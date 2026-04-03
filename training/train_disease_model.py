"""
Fast Disease Detection Training - 8 Distinct Classes
"""
import os, json, random, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "disease_dataset")
MDL = os.path.join(BASE, "models")
MPATH = os.path.join(MDL, "disease_model.h5")
LPATH = os.path.join(MDL, "disease_labels.json")
XPATH = os.path.join(MDL, "disease_metrics.json")

SZ = 224
BS = 16
EP1 = 15
EP2 = 5
NIMG = 80

CLASSES = {
    "Healthy_Leaf":       {"lc": [(45,140,48),(65,165,60)], "spots": False, "noise": 15, "voff": -18},
    "Yellow_Disease":     {"lc": [(170,175,45),(210,205,75)], "spots": False, "noise": 20, "voff": -15},
    "Brown_Spot":         {"lc": [(60,115,42),(80,135,55)], "spots": True, "sc": [(55,38,22),(85,55,32)], "sn":(20,45),"ss":(4,12), "noise": 12, "voff": -15},
    "Rust_Disease":       {"lc": [(58,110,38),(78,130,50)], "spots": True, "sc": [(205,135,48),(240,170,72)], "sn":(30,70),"ss":(2,6), "noise": 10, "voff": -15},
    "Black_Rot":          {"lc": [(52,100,35),(72,120,48)], "spots": True, "sc": [(22,18,14),(42,32,22)], "sn":(8,18),"ss":(12,32), "noise": 10, "voff": -18},
    "Powdery_Mildew":     {"lc": [(55,120,42),(75,140,55)], "spots": True, "sc": [(215,215,210),(240,240,235)], "sn":(40,90),"ss":(3,9), "noise": 12, "voff": -12},
    "Mosaic_Virus":       {"lc": [(48,125,42),(68,148,58)], "spots": True, "sc": [(28,85,28),(78,155,58)], "sn":(25,45),"ss":(8,22), "noise": 18, "voff": -10},
    "Wilting_Leaf":       {"lc": [(105,85,48),(145,118,72)], "spots": True, "sc": [(78,58,32),(128,98,52)], "sn":(10,25),"ss":(5,14), "noise": 25, "voff": -25},
}

def gen_img(cfg, sz=SZ):
    mask = Image.new('L',(sz,sz),0)
    d = ImageDraw.Draw(mask)
    cx,cy=sz//2,sz//2
    rx=random.randint(int(sz*0.32),int(sz*0.42))
    ry=random.randint(int(rx*0.48),int(rx*0.62))
    pts=[]
    for a in range(0,360,3):
        r=math.radians(a)
        rm=1.0+0.1*math.sin(r*random.randint(3,7))
        pts.append((cx+int(rx*rm*math.cos(r)),cy+int(ry*rm*math.sin(r))))
    d.polygon(pts,fill=255)
    mask=mask.filter(ImageFilter.GaussianBlur(2))
    m=np.array(mask).astype(np.float32)/255.0

    l1,l2=cfg["lc"]
    br,bg,bb=random.uniform(l1[0],l2[0]),random.uniform(l1[1],l2[1]),random.uniform(l1[2],l2[2])
    nv=cfg.get("noise",15)
    yy,xx=np.mgrid[0:sz,0:sz].astype(np.float32)
    dist=np.sqrt((xx-cx)**2+(yy-cy)**2)
    md=np.sqrt(cx**2+cy**2)
    gr=1.0-0.1*(dist/md)
    img=np.stack([br*gr+np.random.normal(0,nv,(sz,sz)),
                  bg*gr+np.random.normal(0,nv,(sz,sz)),
                  bb*gr+np.random.normal(0,nv,(sz,sz))],axis=-1).astype(np.float32)

    if cfg["spots"]:
        ns=random.randint(cfg["sn"][0],cfg["sn"][1])
        for _ in range(ns):
            sx=random.randint(int(sz*0.1),int(sz*0.9))
            sy=random.randint(int(sz*0.1),int(sz*0.9))
            sr=random.randint(cfg["ss"][0],cfg["ss"][1])
            sc=random.choice(cfg["sc"])
            yl,yh=max(0,sy-sr),min(sz,sy+sr+1)
            xl,xh=max(0,sx-sr),min(sz,sx+sr+1)
            Y,X=np.meshgrid(np.arange(yl,yh),np.arange(xl,xh),indexing='ij')
            ci=np.clip(1.0-np.sqrt((X-sx)**2+(Y-sy)**2)/sr,0,1)
            for c in range(3):
                img[yl:yh,xl:xh,c]=img[yl:yh,xl:xh,c]*(1-ci*0.8)+sc[c]*ci*0.8

    vo=cfg.get("voff",-20)
    vc=(max(0,br+vo),max(0,bg+vo),max(0,bb+vo))
    vi=Image.fromarray(np.clip(img,0,255).astype(np.uint8))
    vd=ImageDraw.Draw(vi)
    vci=tuple(int(v) for v in vc)
    vd.line([(cx,cy-ry+10),(cx,cy+ry-10)],fill=vci,width=2)
    for i in range(4):
        sy=cy-ry+25+i*(2*ry-50)//4
        for s in [-1,1]:
            vd.line([(cx,sy),(cx+s*(rx-15),sy+random.randint(-10,10))],fill=vci,width=1)

    img=np.array(vi).astype(np.float32)
    for c in range(3): img[:,:,c]*=m
    inv=1.0-m
    bgc=np.array([random.randint(170,195),random.randint(175,200),random.randint(180,205)],dtype=np.float32)
    for c in range(3): img[:,:,c]+=bgc[c]*inv
    out=Image.fromarray(np.clip(img,0,255).astype(np.uint8))
    out=out.filter(ImageFilter.GaussianBlur(0.6))
    return out

def gen_data():
    print(f"[INFO] Generating {NIMG} images x {len(CLASSES)} classes...")
    tot=0
    for i,(nm,cfg) in enumerate(CLASSES.items()):
        cd=os.path.join(DATA,nm)
        os.makedirs(cd,exist_ok=True)
        ex=len([f for f in os.listdir(cd) if f.endswith('.jpg')])
        if ex>=NIMG:
            print(f"  [{i+1}/{len(CLASSES)}] {nm}: {ex} exist")
            tot+=ex; continue
        for j in range(NIMG):
            gen_img(cfg,SZ).save(os.path.join(cd,f"{nm}_{j:04d}.jpg"),quality=90)
        tot+=NIMG
        print(f"  [{i+1}/{len(CLASSES)}] {nm}: {NIMG} generated")
    print(f"[INFO] Total: {tot} images")

def main():
    print("="*55)
    print("  DISEASE DETECTION - 8 CLASS FAST TRAINING")
    print("="*55)
    gen_data()

    print("\n[INFO] Creating generators...")
    tg=ImageDataGenerator(rescale=1/255,rotation_range=25,width_shift_range=0.15,
        height_shift_range=0.15,zoom_range=0.15,horizontal_flip=True,vertical_flip=True,
        brightness_range=[0.75,1.25],fill_mode="nearest",validation_split=0.2
    ).flow_from_directory(DATA,target_size=(SZ,SZ),batch_size=BS,
        class_mode="categorical",subset="training",shuffle=True)

    vg=ImageDataGenerator(rescale=1/255,validation_split=0.2
    ).flow_from_directory(DATA,target_size=(SZ,SZ),batch_size=BS,
        class_mode="categorical",subset="validation",shuffle=False)

    nc=tg.num_classes
    ci=tg.class_indices
    i2l={v:k for k,v in ci.items()}
    print(f"[INFO] Classes:{nc} Train:{tg.samples} Val:{vg.samples}")

    print("[INFO] Building MobileNetV2...")
    base=MobileNetV2(weights="imagenet",include_top=False,input_shape=(SZ,SZ,3))
    base.trainable=False
    inp=Input(shape=(SZ,SZ,3))
    x=base(inp,training=False)
    x=GlobalAveragePooling2D()(x)
    x=BatchNormalization()(x)
    x=Dense(128,activation="relu")(x)
    x=Dropout(0.4)(x)
    x=Dense(64,activation="relu")(x)
    x=Dropout(0.3)(x)
    out=Dense(nc,activation="softmax")(x)
    model=Model(inp,out)
    params=model.count_params()
    print(f"[INFO] Params: {params:,}")

    model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
    os.makedirs(MDL,exist_ok=True)
    cbs=[EarlyStopping(monitor="val_accuracy",patience=6,restore_best_weights=True,verbose=1),
         ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6,verbose=1),
         ModelCheckpoint(MPATH,monitor="val_accuracy",save_best_only=True,verbose=0)]

    print(f"\n[INFO] Phase 1: Frozen base ({EP1} epochs)")
    h1=model.fit(tg,epochs=EP1,validation_data=vg,callbacks=cbs,verbose=1)

    print(f"\n[INFO] Phase 2: Fine-tuning last 30 layers ({EP2} epochs)")
    base.trainable=True
    for l in base.layers[:-30]: l.trainable=False
    model.compile(optimizer=Adam(1e-4),loss="categorical_crossentropy",metrics=["accuracy"])
    h2=model.fit(tg,epochs=EP2,validation_data=vg,callbacks=cbs,verbose=1)

    vl,va=model.evaluate(vg,verbose=0)
    with open(LPATH,"w") as f: json.dump(i2l,f,indent=2)
    m={"val_accuracy":round(va,4),"val_loss":round(vl,4),"num_classes":nc,
       "total_params":int(params),"epochs_trained":len(h1.history["accuracy"])+len(h2.history["accuracy"]),
       "image_size":SZ,"class_names":list(ci.keys())}
    with open(XPATH,"w") as f: json.dump(m,f,indent=2)

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"  Validation Accuracy: {va*100:.2f}%")
    print(f"  Validation Loss:     {vl:.4f}")
    print(f"  Classes:             {nc}")
    print(f"  Epochs:              {m['epochs_trained']}")
    print(f"{'='*55}")
    print("[DONE]")

if __name__=="__main__":
    main()
