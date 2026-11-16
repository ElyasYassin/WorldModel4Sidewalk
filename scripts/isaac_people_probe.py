"""Probe an NVIDIA character + walk animation: skeleton / animation joints, time samples, facing direction, units."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdGeom, UsdSkel  # noqa: E402

R = get_assets_root_path()
stage = omni.usd.get_context().get_stage()
import sys as _s
URLS = ["Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
        "Isaac/People/Characters/original_male_adult_construction_01/male_adult_construction_01.usd",
        "Isaac/People/Characters/original_female_adult_business_02/female_adult_business_02.usd"]
for k, u in enumerate(URLS):
    c = stage.DefinePrim(f"/World/C{k}/asset", "Xform")
    c.GetReferences().AddReference(f"{R}/{u}")
ch = stage.DefinePrim("/World/C/asset", "Xform")
ch.GetReferences().AddReference(f"{R}/Isaac/People/Characters/F_Business_02/F_Business_02.usd")
an = stage.DefinePrim("/World/A")
an.GetReferences().AddReference(f"{R}/Isaac/People/Animations/stand_walk_loop_in_place.skelanim.usd")
for _ in range(10):
    app.update()
print("[pp] stage mpu", UsdGeom.GetStageMetersPerUnit(stage), flush=True)
for k in range(len(URLS)):
    for q in Usd.PrimRange(stage.GetPrimAtPath(f"/World/C{k}")):
        if q.IsA(UsdSkel.Skeleton):
            j = UsdSkel.Skeleton(q).GetJointsAttr().Get()
            print(f"[pp] {URLS[k].split('/')[-1]} skel joints {len(j)} {list(j)[:4]}", flush=True)
            break
for q in Usd.PrimRange(stage.GetPrimAtPath("/World/C")):
    if q.IsA(UsdSkel.Root) or q.IsA(UsdSkel.Skeleton) or q.IsA(UsdSkel.Animation):
        print("[pp] char prim", q.GetPath(), q.GetTypeName(), flush=True)
    if q.IsA(UsdSkel.Skeleton):
        sk = UsdSkel.Skeleton(q)
        j = sk.GetJointsAttr().Get()
        print("[pp] skel joints", len(j), list(j)[:6], flush=True)
        for i, n in enumerate(j):
            print(f"[pp] J {i} {n}  pos {np.array(xf_all[i].ExtractTranslation()).round(3) if False else ''}", flush=True)
        rel = UsdSkel.BindingAPI(q).GetAnimationSourceRel()
        print("[pp] skel animSource", rel.GetTargets() if rel else None, flush=True)
        rest = sk.GetRestTransformsAttr().Get()
        cache = UsdSkel.Cache()
        sq = cache.GetSkelQuery(sk)
        xf = sq.ComputeJointSkelTransforms(Usd.TimeCode.Default())
        names = [str(n).split("/")[-1] for n in j]
        for key in ("l_toe", "l_ball", "l_foot", "l_ankle", "head", "pelvis", "Hips", "LeftToeBase", "LeftFoot"):
            for i, n in enumerate(names):
                if key.lower() in n.lower():
                    print(f"[pp] joint {n}: {np.array(xf[i].ExtractTranslation()).round(3)}", flush=True)
                    break
for q in Usd.PrimRange(an):
    if q.IsA(UsdSkel.Animation):
        A = UsdSkel.Animation(q)
        print("[pp] anim prim", q.GetPath(), "joints", len(A.GetJointsAttr().Get()), list(A.GetJointsAttr().Get())[:4], flush=True)
        ts = A.GetRotationsAttr().GetTimeSamples()
        print("[pp] anim rot samples", len(ts), ts[:3], ts[-2:], "trans samples", len(A.GetTranslationsAttr().GetTimeSamples()), flush=True)
print("[pp] anim layer tcps", an.GetStage().GetTimeCodesPerSecond(), flush=True)
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
print("[pp] char bbox", bb.ComputeWorldBound(stage.GetPrimAtPath("/World/C")).ComputeAlignedRange(), flush=True)
app.close()
