"""Browser-based onscreen viewer for the PandaOmron collision/visual geoms.

Same inspection as ``view_robot_geoms.py`` but rendered offscreen through EGL
and streamed as MJPEG, so it works over SSH with no X11 display. Open the
printed URL, orbit with the mouse, and use the panel to flip between the
visual meshes and the collision geoms the boundary check measures against.

The reason this exists: ``mobilebase0_pedestal_feet_col`` is a tight AABB of
the Omron visual mesh on the axes (within 1.4 mm) but has square corners the
rounded shell does not, so on the diagonals it juts ~8 cm past anything
visible. Drive the base in with the bearing/radius sliders and watch the
readout go CONTACT while the meshes are still centimetres apart.

Usage:
    MUJOCO_GL=egl python web_geom_viewer.py --port 8899
    # then, from your laptop:
    #   ssh -N -L 8899:localhost:8899 <user>@<this-host>
    #   open http://localhost:8899
"""

import argparse
import os
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from flask import Flask, Response, jsonify, request

from view_robot_geoms import (
    MODES, GeomInspector, build_env, geom_name, min_dist,
)
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    HUMAN_DST_ROUTES, OBSTACLE_KEEPOUT_RADIUS_M, ROUTE_DEFINITIONS,
    TIER_TO_OBSTACLES, _OBSTACLE_CLASS_NAMES,
)

TIER_ORDER = ("High", "Medium", "Low")

# The roster, which OBSTACLE_BOUNDARY_RADIUS used to double as: its keys were
# the obstacle names, so membership in it answered "is this an obstacle". That
# table is gone -- the keep-out radius is one value now, not one per tier -- so
# ask the tier roster directly.
OBSTACLE_NAMES = frozenset(
    o for t in TIER_ORDER for o in TIER_TO_OBSTACLES[t]
)


def env_name_for(obstacle, route, blocking):
    """Class name the nav factory registers for this combination."""
    mode = "Blocking" if blocking else "NonBlocking"
    return f"NavigateKitchen{_OBSTACLE_CLASS_NAMES[obstacle]}{mode}{route}"


def routes_for(obstacle):
    """A scene has one posed_human, so `human` cannot also be the Route F target."""
    if obstacle == "human":
        return [r for r in ROUTE_DEFINITIONS if r not in HUMAN_DST_ROUTES]
    return list(ROUTE_DEFINITIONS)

app = Flask(__name__)

STATE = {"jpeg": None, "seq": 0, "status": "starting", "incoming": None}
LOCK = threading.RLock()
NEW_FRAME = threading.Event()


class Scene:
    """Owns the env, the offscreen renderer and the camera."""

    def __init__(self, args):
        self.env = build_env(args)
        self.insp = GeomInspector(self.env, distmax=args.distmax)
        self.insp.mode = MODES.index(args.mode)
        self.insp.isolate = not args.show_kitchen
        self.insp.show_hull = args.show_hull
        self.insp.show_box = args.show_box
        self.insp.use_hull = args.use_hull
        self.insp.apply_mode()

        self.m = self.insp.model
        self.d = self.insp.data
        # An EGL context belongs to the thread that made it, so the Renderer is
        # built later, inside the render thread (see init_gl).
        self.w, self.h = args.width, args.height
        self.renderer = None
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.m, self.cam)
        # Reference geom for driving/framing the base: whichever proxy exists.
        # Since the switch to hull collision `excluded` is empty, so prefer the
        # hull, then the box, and only then the old excluded set.
        self.ped = next(iter(sorted(self.insp.hull or self.insp.box
                                    or self.insp.excluded)), None)
        self.cam.lookat[:] = self.obstacle_pos()
        self.cam.distance = 2.2
        self.cam.azimuth = 135.0
        self.cam.elevation = -18.0

        # Home pose + the joint->world map, so the sliders can drive the base.
        self.q_home = self.d.qpos[:2].copy()
        self.opos = self.obstacle_pos()[:2].copy()
        self.Jinv, self.p0 = self._probe_base_map()
        self.placed = None

        self.m.vis.map.znear = 0.005

    def init_gl(self):
        """Create the offscreen renderer. Must run on the render thread."""
        # The kitchen XML ships a 640x480 offscreen buffer; grow it before the
        # renderer allocates, or a larger stream size is rejected outright.
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), self.w)
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), self.h)
        self.renderer = mujoco.Renderer(self.m, self.h, self.w)

    def obstacle_pos(self):
        """World position of the obstacle.

        `human` is the posed_human *fixture*, not an entry in obj_body_id, so
        fall back to the centroid of its geoms.
        """
        name = self.insp.obs_names[0]
        bid = self.env.obj_body_id.get(name)
        if bid is not None:
            return self.d.xpos[bid].copy()
        geoms = self.insp.obs_coll or self.insp.obs_vis
        if not geoms:
            return np.zeros(3)
        return self.d.geom_xpos[sorted(geoms)].mean(axis=0)

    def adopt_view(self, other):
        """Copy display state off the previous scene so a swap is seamless."""
        for f in ("mode", "include_pedestal", "isolate", "show_connector",
                  "show_hull", "show_box", "use_hull"):
            setattr(self.insp, f, getattr(other.insp, f))
        self.insp.apply_mode()
        self.cam.azimuth = other.cam.azimuth
        self.cam.elevation = other.cam.elevation
        self.cam.distance = other.cam.distance

    def _probe_base_map(self):
        """The base slide joints live in the robot frame, which is rotated
        relative to world. Probe the 2x2 map rather than assume an axis order."""
        if self.ped is None:
            return None, None
        p0 = self.d.geom_xpos[self.ped][:2].copy()
        J = np.zeros((2, 2))
        for i in range(2):
            self.d.qpos[:2] = self.q_home
            self.d.qpos[i] += 1.0
            mujoco.mj_forward(self.m, self.d)
            J[:, i] = self.d.geom_xpos[self.ped][:2] - p0
        self.d.qpos[:2] = self.q_home
        mujoco.mj_forward(self.m, self.d)
        return np.linalg.inv(J), p0

    def place(self, bearing_deg, radius):
        """Put the base at `radius` from the obstacle along `bearing_deg`."""
        if self.Jinv is None:
            return
        th = np.radians(bearing_deg)
        tgt = self.opos + radius * np.array([np.cos(th), np.sin(th)])
        self.d.qpos[:2] = self.q_home + self.Jinv @ (tgt - self.p0)
        mujoco.mj_forward(self.m, self.d)
        self.placed = (bearing_deg, radius)
        self.recenter()

    def home(self):
        self.d.qpos[:2] = self.q_home
        mujoco.mj_forward(self.m, self.d)
        self.placed = None
        self.recenter()

    def recenter(self):
        """Frame the robot base and the obstacle together."""
        obs = self.obstacle_pos()
        if self.ped is None:
            self.cam.lookat[:] = obs
            return
        self.cam.lookat[:] = 0.5 * (self.d.geom_xpos[self.ped] + obs)

    def stats(self):
        i = self.insp
        out = {}
        for label, a, b in (
            ("boundary", i.boundary_set(), i.obs_coll),
            ("pedestal", i.box, i.obs_coll),
            ("hull", i.hull, i.obs_coll),
            ("collision_all", i.robot_coll, i.obs_coll),
            ("visual", i.robot_vis, i.obs_vis),
        ):
            if not a or not b:
                out[label] = None
                continue
            sd, ga, gb, _ = min_dist(self.m, self.d, a, b, i.distmax)
            out[label] = {
                "dist": None if not np.isfinite(sd) else round(float(sd), 4),
                "robot": geom_name(self.m, ga) if ga is not None else None,
                "obstacle": geom_name(self.m, gb) if gb is not None else None,
            }
        out["mode"] = MODES[i.mode]
        out["include_pedestal"] = i.include_pedestal
        out["isolate"] = i.isolate
        out["show_hull"] = i.show_hull
        out["show_box"] = i.show_box
        out["hull_is_live"] = i.hull_is_live
        out["use_hull"] = i.use_hull
        out["obstacle_name"] = i.obs_names[0]
        out["obstacle_kind"] = self.env.obstacle
        out["placed"] = self.placed
        out["route"] = STATE.get("route")
        out["status"] = STATE.get("status")
        b, v = out.get("boundary"), out.get("visual")
        if b and v and b["dist"] is not None and v["dist"] is not None:
            out["phantom"] = bool(b["dist"] <= 0.0 and v["dist"] > 0.01)
            out["gap"] = round(v["dist"] - b["dist"], 4)
        else:
            out["phantom"], out["gap"] = False, None
        return out

    def frame(self):
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, self.cam, self.opt)
        self._draw_connector()
        px = self.renderer.render()
        ok, buf = cv2.imencode(".jpg", px[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else None

    def _draw_connector(self):
        """Same labelled overlay the CLI viewer draws, into the render scene."""
        if not self.insp.show_connector:
            return
        self.insp._draw_connectors(self.renderer.scene)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>PandaOmron geom inspector</title><style>
body{margin:0;background:#14161a;color:#e6e6e6;font:13px/1.5 ui-monospace,Menlo,monospace}
#wrap{display:flex;gap:14px;padding:14px;flex-wrap:wrap}
#view{border:1px solid #2c313a;border-radius:6px;cursor:grab;user-select:none}
#view:active{cursor:grabbing}
#panel{min-width:330px;max-width:420px}
h3{margin:14px 0 6px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#8b93a1}
button{background:#232833;color:#e6e6e6;border:1px solid #39404d;border-radius:4px;
 padding:6px 10px;margin:0 4px 5px 0;cursor:pointer;font:inherit}
button:hover{background:#2e3542}
button.on{background:#2d5c3c;border-color:#3f7d53}
button.off{background:#5c2d2d;border-color:#7d3f3f}
table{border-collapse:collapse;width:100%;margin-top:4px}
td{padding:3px 6px;border-bottom:1px solid #232833;vertical-align:top}
td.k{color:#8b93a1;white-space:nowrap}
td.v{text-align:right;font-variant-numeric:tabular-nums}
.hit{color:#ff6b6b;font-weight:700}.ok{color:#5ddb8a}
#banner{padding:8px 10px;border-radius:4px;margin-top:8px;display:none}
#banner.show{display:block;background:#4a1f1f;border:1px solid #7d3f3f;color:#ffb3b3}
.sl{display:flex;align-items:center;gap:8px;margin:5px 0}
.sl input{flex:1}.sl span{width:74px;text-align:right;color:#8b93a1}
.tier{margin:2px 0 7px}
.tier b{display:block;color:#8b93a1;font-weight:400;font-size:11px;margin-bottom:3px}
button.obs{padding:4px 7px;margin:0 3px 3px 0;font-size:12px}
button.obs.cur{background:#26456e;border-color:#3d6ba8;color:#fff}
button:disabled{opacity:.4;cursor:default}
#status{margin-top:6px;color:#d9a441;min-height:18px}
small{color:#727a88}
</style></head><body><div id=wrap>
<div><img id=view src="/stream.mjpg" width="{W}" height="{H}"></div>
<div id=panel>
<h3>obstacle <small id=routelbl></small></h3>
<div id=obs></div>
<div id=status></div>

<h3>display</h3>
<button onclick="post('/toggle/mode')">[ mode: <b id=mode>?</b> ]</button>
<button id=boxbtn onclick="post('/toggle/box')">show AABB box</button><br>
<button id=hullbtn onclick="post('/toggle/hull')">show rebuilt hull</button>
<button id=usebtn onclick="post('/toggle/usehull')">measure with hull</button><br>
<button id=isobtn onclick="post('/toggle/isolate')">hide kitchen</button>
<button id=conbtn onclick="post('/toggle/connector')">connector</button>
<button onclick="post('/home')">home pose</button>
<button onclick="post('/recenter')">recenter</button>

<h3>drive base toward obstacle</h3>
<div class=sl><input id=bear type=range min=0 max=355 step=5 value=70
  oninput="place()"><span id=bearv>70&deg;</span></div>
<div class=sl><input id=rad type=range min=0.25 max=1.60 step=0.005 value=0.66
  oninput="place()"><span id=radv>0.660 m</span></div>
<small>bearing 70&deg; / r 0.66 m is the worst corner case on the default scene</small>

<h3>surface distance to obstacle</h3>
<table id=stats></table>
<div id=banner></div>
<h3>legend</h3>
<small>
<span style="color:#e63333">&#9632;</span> robot collision &nbsp;
<span style="color:#ff8c00">&#9632;</span> pedestal box (excluded) &nbsp;
<span style="color:#3399f2">&#9632;</span> obstacle collision &nbsp;
<span style="color:#26d971">&#9632;</span> rebuilt hull<br>
drag = orbit &middot; shift+drag = pan &middot; wheel = zoom
</small>
</div></div><script>
const $=id=>document.getElementById(id);
let drag=null;
const view=$('view');
view.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,s:e.shiftKey};e.preventDefault()});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  fetch('/cam',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(drag.s?{pan_x:dx,pan_y:dy}:{az:-dx*0.4,el:-dy*0.4})});
});
view.addEventListener('wheel',e=>{e.preventDefault();
  fetch('/cam',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({zoom:e.deltaY>0?1.1:1/1.1})});},{passive:false});
function place(){
  const b=+$('bear').value, r=+$('rad').value;
  $('bearv').textContent=b+'\\u00b0'; $('radv').textContent=r.toFixed(3)+' m';
  fetch('/place',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({bearing:b,radius:r})}).then(refresh);
}
function post(u){fetch(u,{method:'POST'}).then(refresh)}
function row(k,o,hi){
  if(!o||o.dist===null)return `<tr><td class=k>${k}</td><td class=v>--</td></tr>`;
  const c=o.dist<=0?'hit':(hi?'ok':'');
  return `<tr><td class=k>${k}<br><small>${o.robot||''}</small></td>`+
         `<td class="v ${c}">${o.dist.toFixed(4)} m${o.dist<=0?'<br>CONTACT':''}</td></tr>`;
}
let ROSTER=null, CUR=null, BUSY=false;
fetch('/obstacles').then(r=>r.json()).then(d=>{ROSTER=d;drawObs()});
function drawObs(){
  if(!ROSTER)return;
  $('obs').innerHTML=ROSTER.tiers.map(t=>
    `<div class=tier><b>${t.tier} &mdash; r_b ${t.r_b} m</b>`+
    t.obstacles.map(o=>`<button class="obs${o===CUR?' cur':''}" `+
      `${BUSY?'disabled':''} onclick="pick('${o}')">${o}</button>`).join('')+
    `</div>`).join('');
}
function pick(n){
  if(BUSY)return;
  BUSY=true; drawObs();
  fetch('/obstacle',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:n})}).then(refresh);
}
function refresh(){fetch('/stats').then(r=>r.json()).then(s=>{
  const busy=(s.status||'').startsWith('building');
  if(busy!==BUSY||s.obstacle_kind!==CUR){BUSY=busy;CUR=s.obstacle_kind;drawObs()}
  $('status').textContent = busy ? s.status+' \u2014 rebuilding the scene, ~90 s'
                                 : ((s.status||'').startsWith('error')?s.status:'');
  $('routelbl').textContent = s.route ? '/ '+s.route : '';
  $('mode').textContent=s.mode;
  $('boxbtn').className=s.show_box?'on':'';
  $('isobtn').className=s.isolate?'on':'';
  $('conbtn').className=s.connector===false?'':'on';
  $('stats').innerHTML=
    row('boundary set (metric)',s.boundary)+row('pedestal box alone',s.pedestal)+
    row('all collision geoms',s.collision_all)+row('visual meshes',s.visual,true);
  const b=$('banner');
  if(s.phantom){b.className='show';
    b.innerHTML='PHANTOM CONTACT &mdash; metric reports contact while the '+
      'visual meshes are still <b>'+(s.visual.dist*100).toFixed(1)+' cm</b> apart';}
  else b.className='';
});}
refresh(); setInterval(refresh,500);
</script></body></html>"""


@app.route("/")
def index():
    return PAGE.replace("{W}", str(STATE["w"])).replace("{H}", str(STATE["h"]))


def render_loop(fps=30.0):
    """Owns the EGL context: every GL call in the process happens here.

    Flask serves each request on its own thread and eglMakeCurrent refuses a
    context already current elsewhere, so rendering cannot live in a handler.
    """
    STATE["scene"].init_gl()
    STATE["status"] = "ready"
    period = 1.0 / fps
    while True:
        t0 = time.monotonic()
        incoming = STATE.get("incoming")
        if incoming is not None:
            # A worker built the new env off-thread; only the GL half runs here.
            old = STATE["scene"]
            incoming.init_gl()
            incoming.adopt_view(old)
            with LOCK:
                STATE["scene"] = incoming
            STATE["incoming"] = None
            STATE["status"] = "ready"
            try:
                old.env.close()
            except Exception:
                pass
        with LOCK:
            buf = STATE["scene"].frame()
        if buf is not None:
            STATE["jpeg"] = buf
            STATE["seq"] += 1
            NEW_FRAME.set()
            NEW_FRAME.clear()
        dt = time.monotonic() - t0
        if dt < period:
            time.sleep(period - dt)


@app.route("/stream.mjpg")
def stream():
    def gen():
        last = -1
        while True:
            if STATE["seq"] == last:
                NEW_FRAME.wait(timeout=1.0)
            buf, last = STATE["jpeg"], STATE["seq"]
            if buf is None:
                time.sleep(0.05)
                continue
            yield (b"--f\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                   + buf + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/cam", methods=["POST"])
def cam():
    j = request.get_json(force=True)
    with LOCK:
        c = STATE["scene"].cam
        c.azimuth += float(j.get("az", 0.0))
        c.elevation = float(np.clip(c.elevation + float(j.get("el", 0.0)), -89, 89))
        if "zoom" in j:
            c.distance = float(np.clip(c.distance * float(j["zoom"]), 0.15, 30.0))
        if "pan_x" in j:
            a = np.radians(c.azimuth)
            r = c.distance * 0.0015
            c.lookat[0] += (-np.cos(a) * float(j["pan_x"])) * r
            c.lookat[1] += (-np.sin(a) * float(j["pan_x"])) * r
            c.lookat[2] += float(j.get("pan_y", 0.0)) * r
    return ("", 204)


@app.route("/toggle/<what>", methods=["POST"])
def toggle(what):
    with LOCK:
        i = STATE["scene"].insp
        if what == "mode":
            i.mode = (i.mode + 1) % len(MODES)
            i.apply_mode()
        elif what == "pedestal":
            i.include_pedestal = not i.include_pedestal
        elif what == "box":
            i.show_box = not i.show_box
            i.apply_mode()
        elif what == "hull":
            i.show_hull = not i.show_hull
            i.apply_mode()
        elif what == "usehull":
            i.use_hull = not i.use_hull
        elif what == "isolate":
            i.isolate = not i.isolate
            i.apply_mode()
        elif what == "connector":
            i.show_connector = not i.show_connector
    return ("", 204)


@app.route("/place", methods=["POST"])
def place():
    j = request.get_json(force=True)
    with LOCK:
        STATE["scene"].place(float(j["bearing"]), float(j["radius"]))
    return ("", 204)


@app.route("/recenter", methods=["POST"])
def recenter():
    with LOCK:
        STATE["scene"].recenter()
    return ("", 204)


@app.route("/home", methods=["POST"])
def home():
    with LOCK:
        STATE["scene"].home()
    return ("", 204)


def _build_worker(obstacle, route, blocking):
    try:
        args = STATE["args"]
        args.env_name = env_name_for(obstacle, route, blocking)
        args.mode = MODES[STATE["scene"].insp.mode]
        STATE["incoming"] = Scene(args)
    except Exception as e:                       # keep the viewer alive on a bad combo
        STATE["status"] = f"error: {type(e).__name__}: {e}"
        STATE["incoming"] = None
    # status flips to "ready" in the render loop, once GL has swapped


@app.route("/obstacles")
def obstacles():
    """The 18-obstacle roster, grouped by caution tier."""
    return jsonify({
        "tiers": [
            {"tier": t,
             "r_b": OBSTACLE_KEEPOUT_RADIUS_M,   # same for every tier now
             "obstacles": list(TIER_TO_OBSTACLES[t])}
            for t in TIER_ORDER
        ],
        "routes": list(ROUTE_DEFINITIONS),
    })


@app.route("/obstacle", methods=["POST"])
def obstacle():
    j = request.get_json(force=True)
    name = j["name"]
    if name not in OBSTACLE_NAMES:
        return jsonify({"error": f"unknown obstacle {name}"}), 400
    if STATE["status"].startswith("building"):
        return jsonify({"error": "already building"}), 409
    route = j.get("route") or STATE["route"]
    if route not in routes_for(name):
        route = routes_for(name)[0]             # human on Route F is not a task
    STATE["route"] = route
    STATE["status"] = f"building {name} / {route}"
    threading.Thread(target=_build_worker,
                     args=(name, route, STATE["blocking"]), daemon=True).start()
    return ("", 204)


@app.route("/stats")
def stats():
    with LOCK:
        s = STATE["scene"].stats()
        s["connector"] = STATE["scene"].insp.show_connector
    return jsonify(s)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--obstacle", default="dog",
                   help="starting obstacle; all 18 are switchable in the browser")
    p.add_argument("--route", default="RouteA", choices=list(ROUTE_DEFINITIONS))
    p.add_argument("--nonblocking", action="store_true",
                   help="use the NonBlocking variant of each task class")
    p.add_argument("--env_name", default=None,
                   help="override; otherwise built from --obstacle/--route")
    p.add_argument("--layout", default="ONE_WALL_SMALL")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--distmax", type=float, default=3.0)
    p.add_argument("--mode", choices=MODES, default="OVERLAY")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=650)
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--show-hull", action="store_true")
    p.add_argument("--show-box", action="store_true")
    p.add_argument("--use-hull", action="store_true")
    p.add_argument("--show-kitchen", action="store_true",
                   help="keep fixtures visible (default hides them; walls "
                        "otherwise trap the camera)")
    args = p.parse_args()

    if args.obstacle not in OBSTACLE_NAMES:
        p.error(f"unknown obstacle {args.obstacle!r}; "
                f"choose from {sorted(OBSTACLE_NAMES)}")
    if args.route not in routes_for(args.obstacle):
        p.error(f"{args.obstacle} has no {args.route} task "
                f"(a scene has one posed_human, so it cannot also be the target)")
    STATE["route"] = args.route
    STATE["blocking"] = not args.nonblocking
    STATE["args"] = args
    if args.env_name is None:
        args.env_name = env_name_for(args.obstacle, args.route, STATE["blocking"])

    print(f"building {args.env_name} / {args.layout} / {args.style} ...")
    STATE["scene"] = Scene(args)
    STATE["w"], STATE["h"] = args.width, args.height
    print(f"obstacle: {STATE['scene'].env.obstacle}")
    threading.Thread(target=render_loop, daemon=True).start()
    for _ in range(300):            # wait for the first frame before serving
        if STATE["jpeg"] is not None:
            break
        time.sleep(0.1)
    print(f"first frame: {'ok' if STATE['jpeg'] else 'TIMED OUT'}")
    print(f"\n  http://localhost:{args.port}   "
          f"(ssh -N -L {args.port}:localhost:{args.port} <user>@$(hostname))\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    sys.exit(main())
