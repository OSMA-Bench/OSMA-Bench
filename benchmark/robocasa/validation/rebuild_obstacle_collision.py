"""Regenerate an obstacle's VHACD collision proxy so it hugs its visual mesh.

The shipped proxies were produced with VHACD's defaults (32 hulls, 100k
voxels, 1% volume error). Those hulls bulge well past the surface they stand
in for: on the dog the collision union overshoots the visual silhouette by
19.0 mm mean / 25.4 mm max at asset scale. That is what makes the boundary
check report a touch while the rendered surfaces are still centimetres apart.

Raising the voxel resolution matters far more than raising the hull count --
128 hulls at 8M voxels beats 256 hulls at 2M, at two thirds the geometry:

    setting                       hulls  verts   mean    max   (mm)
    shipped (h32 r100k e1)           32   1008  18.96  25.36
    h64  r400k e0.5                  64   2067  11.94  16.45
    h128 r1M   e0.2                 128   3877   9.03  11.93
    h256 r2M   e0.1                 256   6534   7.21   9.54
    h128 r8M   e0.05 l1             128   5641   4.34   6.04   <- default here
    h256 r8M   e0.05 l1             256   8432   4.51   6.04

Writes the new pieces into <asset>/collision/ and rewrites model.xml to point
at them. The previous collision/ and model.xml are moved aside first.

Usage:
    python rebuild_obstacle_collision.py --asset .../lrs_objs/dog
    python rebuild_obstacle_collision.py --asset .../dog --hulls 256 --dry-run
    python rebuild_obstacle_collision.py --asset .../dog --restore
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

VHACD = "/usr/local/bin/TestVHACD"
BACKUP_SUFFIX = ".orig"


def load_pieces(decomp_obj):
    """VHACD writes every hull into one .obj as separate objects."""
    sc = trimesh.load(decomp_obj, process=False, split_object=True,
                      group_material=False)
    if isinstance(sc, trimesh.Scene):
        return list(sc.geometry.values())
    return sc.split(only_watertight=False)


def overshoot(pieces, visual, n_dirs=20000, seed=0):
    """How far the collision union reaches past the visual silhouette.

    Support-function difference, not surface distance: pieces that fill the
    model's hollow interior are harmless and must not count as inflation.
    """
    V = np.asarray(visual.vertices)
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n_dirs, 3))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    P = np.vstack([np.asarray(p.vertices) for p in pieces])
    return (P @ D.T).max(axis=0) - (V @ D.T).max(axis=0)


def visual_obj(asset):
    objs = sorted(glob.glob(os.path.join(asset, "visual", "*.obj")))
    if not objs:
        raise SystemExit(f"no visual/*.obj under {asset}")
    return objs[0]


def restore(asset):
    for rel in ("collision", "model.xml"):
        src = os.path.join(asset, rel + BACKUP_SUFFIX)
        dst = os.path.join(asset, rel)
        if not os.path.exists(src):
            print(f"  no backup for {rel}")
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f"  restored {rel}")


def rewrite_xml(xml_path, prefix, n_pieces):
    """Swap the collision mesh assets and geoms for the new piece list.

    Visual geoms, materials, textures and sites are left exactly as they were;
    only entries whose mesh name ends in `._coll` are replaced.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    body = root.find(".//body[@name='object']")
    if asset is None or body is None:
        raise SystemExit(f"unexpected structure in {xml_path}")

    old_geoms = [g for g in body.findall("geom")
                 if (g.get("mesh") or "").endswith("._coll")]
    if not old_geoms:
        raise SystemExit(f"no ._coll geoms found in {xml_path}")
    template = old_geoms[0]

    for m in list(asset.findall("mesh")):
        if (m.get("name") or "").endswith("._coll"):
            asset.remove(m)
    for g in old_geoms:
        body.remove(g)

    for i in range(n_pieces):
        name = f"{prefix}_{i}._coll"
        ET.SubElement(asset, "mesh", {
            "file": f"collision/{prefix}_{i}.obj",
            "name": name,
            "scale": "1.0 1.0 1.0",
        })
        attrs = dict(template.attrib)
        attrs["mesh"] = name
        ET.SubElement(body, "geom", attrs)

    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--asset", required=True, help="object asset directory")
    p.add_argument("--hulls", type=int, default=128)
    p.add_argument("--resolution", type=int, default=8_000_000)
    p.add_argument("--error", type=float, default=0.05)
    p.add_argument("--min-edge", type=int, default=1)
    p.add_argument("--max-verts", type=int, default=64)
    p.add_argument("--dry-run", action="store_true",
                   help="decompose and report, write nothing")
    p.add_argument("--restore", action="store_true",
                   help="put the .orig backups back and exit")
    args = p.parse_args()

    asset = os.path.abspath(args.asset)
    if args.restore:
        print(f"restoring {asset}")
        restore(asset)
        return 0
    if not os.path.isfile(VHACD):
        raise SystemExit(f"VHACD binary not found at {VHACD}")

    src = visual_obj(asset)
    visual = trimesh.load(src, process=False, force="mesh")
    old = [trimesh.load(f, process=False, force="mesh")
           for f in sorted(glob.glob(os.path.join(asset, "collision", "*.obj")))]
    print(f"asset  : {asset}")
    print(f"visual : {os.path.basename(src)}  "
          f"{len(visual.vertices)} verts / {len(visual.faces)} faces")

    work = os.path.join(asset, "_vhacd_tmp")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cmd = [VHACD, src, "-h", str(args.hulls), "-r", str(args.resolution),
           "-e", str(args.error), "-l", str(args.min_edge),
           "-v", str(args.max_verts), "-s", "true", "-g", "false", "-o", "obj"]
    subprocess.run(cmd, cwd=work, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decomp = os.path.join(work, "decomp.obj")
    if not os.path.exists(decomp):
        raise SystemExit("VHACD produced no decomp.obj")
    new = load_pieces(decomp)

    go, gn = overshoot(old, visual), overshoot(new, visual)
    print(f"\noutward overshoot past the visual silhouette (asset scale, mm)")
    print(f"  {'':8} {'hulls':>6} {'verts':>7} {'mean':>8} {'p95':>8} {'max':>8}")
    for lbl, g, ms in (("before", go, old), ("after", gn, new)):
        print(f"  {lbl:8} {len(ms):6d} {sum(len(m.vertices) for m in ms):7d} "
              f"{g.mean()*1000:8.2f} {np.percentile(g,95)*1000:8.2f} "
              f"{g.max()*1000:8.2f}")
    print(f"  reduction: mean {100*(1-gn.mean()/go.mean()):.1f}%  "
          f"max {100*(1-gn.max()/go.max()):.1f}%")

    if args.dry_run:
        shutil.rmtree(work, ignore_errors=True)
        print("\n--dry-run: nothing written")
        return 0

    coll = os.path.join(asset, "collision")
    xml = os.path.join(asset, "model.xml")
    for path in (coll, xml):
        bak = path + BACKUP_SUFFIX
        if not os.path.exists(bak):
            shutil.move(path, bak)
            print(f"\nbacked up {os.path.basename(path)} -> "
                  f"{os.path.basename(bak)}")
        else:
            print(f"\nbackup {os.path.basename(bak)} already exists, keeping it")
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
    shutil.copy(xml + BACKUP_SUFFIX, xml)
    os.makedirs(coll, exist_ok=True)

    prefix = "collision_piece"
    for i, mesh in enumerate(new):
        mesh.export(os.path.join(coll, f"{prefix}_{i}.obj"))
    rewrite_xml(xml, prefix, len(new))
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {len(new)} pieces to collision/ and rewrote model.xml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
