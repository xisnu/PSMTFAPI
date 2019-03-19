from __future__ import print_function
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import os

iris_path="Data/IRIS/iris.data"
earthquake_path="/media/parthosarothi/OHWR/Dataset/TimeSeries/UCR_TS_Archive_2015/Earthquakes/Earthquakes_"
osuleaf_path="/media/parthosarothi/OHWR/Dataset/TimeSeries/UCR_TS_Archive_2015/OSULeaf/OSULeaf_"

def load_iris_data(iris_csv):
    f=open(iris_csv)
    line=f.readline()
    iris_df=pd.DataFrame()
    while line:
        info=line.strip("\n").split(",")
        if(len(info)==5):
            label=info[-1]
            features=info[:4]
            features.append(label)
            #print(features)
            df = pd.DataFrame([features], columns=['SL', 'SW', 'PL', 'PW', 'class'])
            iris_df=iris_df.append(df)
        line=f.readline()
    return iris_df

def load_and_split_iris(iris_path,split_ratio):
    iris_df = load_iris_data(iris_path)
    class_labels = list(set(iris_df['class']))
    print(class_labels)
    train_set=pd.DataFrame()
    test_set=pd.DataFrame()
    for cl in class_labels:
        class_loc=iris_df.loc[iris_df['class']==cl]
        nb_samples=len(class_loc)
        class_loc=class_loc.sample(nb_samples)
        #print(class_loc)
        test_volume=int(nb_samples*split_ratio)
        train_volume=nb_samples-test_volume
        #print(class_loc.iloc[train_volume:,:])
        train_set=train_set.append(class_loc.iloc[:train_volume,:])
        test_set=test_set.append(class_loc.iloc[train_volume:,:])

    nbtests=len(test_set)
    test_set=test_set.sample(nbtests)

    nbtrain = len(train_set)
    train_set = train_set.sample(nbtrain)
    print("Train samples=%d Test samples=%d"%(nbtrain,nbtests))
    return train_set,test_set,class_labels

def make_one_hot(batch_labels,all_labels):
    total_classes=len(all_labels)
    total_samples=len(batch_labels)
    one_hot=np.zeros([total_samples,total_classes],dtype=float)
    for ts in range(total_samples):
        one_hot[ts][all_labels.index(batch_labels[ts])]=1
    return one_hot

def load_timeseries_data(timeseries_csv,nbclass,class_label_from_0=False):
    f=open(timeseries_csv)
    line=f.readline()
    sequence_length=[]
    labels=[]
    features=[]
    while line:
        info=line.strip("\n").split(",")
        label=info[0]
        label_one_hot=np.zeros([nbclass])
        if(class_label_from_0):
            label_index=int(label)
        else:
            label_index=int(label)-1
        label_one_hot[label_index]=1
        feat=info[1:]
        nbfeatures=len(feat)
        sequence_length.append(nbfeatures)
        labels.append(label_one_hot)
        features.append(feat)
        line=f.readline()
    max_length=max(sequence_length)
    min_length=min(sequence_length)
    print("Maximum sequence length: ",max_length," minimum sequence length: ",min_length)
    return np.asarray(features),np.asarray(labels),sequence_length

def data_frame_to_array(data_frame,dtype):
    try:
        arr=np.asarray(data_frame.values,dtype)
    except:
        print(data_frame)
    return arr

def encode_seq2seq_input(labels,charmap,maxlength=None,align='C',verbose=False):
    #charmap is list of characters, list index is character map
    #charmap must contain <s>, <e> and <p>
    #labels are sequence of target characters separated by space
    #returns N,label_length,Nc as one hot
    total=len(labels)
    Nc=len(charmap)
    one_hot_labels=[]
    aligned_labels=[]
    if(maxlength is not None):
        one_hot_labels=np.zeros([total,maxlength,Nc])
    for t in range(total):
        if(align=='C'):
            this_label="<s> "+labels[t]+" <e>"
        elif(align=='L'):
            this_label="<s> "+labels[t]
        else:
            this_label=labels[t]+" <e>"
        if(verbose):
            print("Reading label ",this_label)
        characters=this_label.split()
        label_length=len(characters)
        if(maxlength is None):
            one_hot = np.zeros([label_length,Nc])
        for i in range(label_length):
            charpos=charmap[characters[i]]
            if(maxlength is None):
                one_hot[i][charpos]=1
            else:
                one_hot_labels[t][i][charpos]=1
        if(maxlength is None):
            one_hot_labels.append(one_hot)
        aligned_labels.append(this_label)
    one_hot_labels=np.asarray(one_hot_labels)
    return one_hot_labels,aligned_labels

def load_character_map(lexfile,key='Character'):
    map={}
    f=open(lexfile)
    line=f.readline()
    i=0
    while line:
        info=line.strip("\n")
        if(key=='Character'):
            map[info]=i
        else:
            map[i]=info
        i=i+1
        line=f.readline()
    return map

def onehot_to_label(onehot,charmap):
    label=""
    if(onehot.ndim==2):
        for v in onehot:
            maxpos=np.argmax(v)
            char=charmap[maxpos]
            label=label+char+" "
    elif(onehot.ndim==1):
        maxpos=np.argmax(onehot)
        label=charmap[maxpos]
    return label

def crop_image(imagemat,threshold=250,savepath=None):
    #find 1st and last non white rows and 1st and last non white columns
    #imagemat is rows x colums => height x width numpy array
    original_w,original_h=imagemat.shape[1],imagemat.shape[0]
    print("original width=%d, height=%d"%(original_w,original_h))
    #finding top and bottom white rows
    top_flag=True
    bottom_flag=True
    last_top_white=0
    last_bottom_white=original_h-1
    for r in range(original_h):
        for c in range(original_w):
            pixval_top=imagemat[r][c]
            pixval_bottom=imagemat[original_h-r-1][c]
            if(top_flag)and(pixval_top<threshold):
                top_flag=False
                last_top_white=r
                #break
            if(bottom_flag)and(pixval_bottom<threshold):
                bottom_flag=False
                last_bottom_white=r
            if (not top_flag) and (not bottom_flag):
                break
        if(not top_flag)and(not bottom_flag):
            break
    print("Top %d Bottom %d"%(last_top_white,last_bottom_white))
    #finding left and right white rows
    left_flag=True
    right_flag=True
    last_left_white=0
    last_right_white=original_w-1
    for c in range(original_w):
        for r in range(original_h):
            pixval_left=imagemat[r][c]
            pixval_right=imagemat[r][original_w-c-1]
            if(left_flag)and(pixval_left<threshold):
                left_flag=False
                last_left_white=c
            if(right_flag)and(pixval_right<threshold):
                right_flag=False
                last_right_white=c
            if(not left_flag)and(not right_flag):
                break
        if (not left_flag) and (not right_flag):
            break
    print("Left %d Right %d" % (last_left_white, last_right_white))
    new_imagemat=imagemat[last_top_white:original_h-last_bottom_white,last_left_white:original_w-last_right_white]
    newimg=Image.fromarray(new_imagemat.astype('uint8')).convert('L')
    w,h=newimg.size[0],newimg.size[1]
    if(savepath is None):
        newimg.save('cropped.jpeg')
    else:
        newimg.save(savepath)
    return w,h

def crop_image_in_dir(imdir):
    #crops all image in imagedir and replace original image
    f=open("Croplog.txt",'w')
    for root,sd,filenames in os.walk(imdir):
        for fname in filenames:
            if(fname[-4:]=='jpeg'):#this is an image file
                abs_filename=os.path.join(root,fname)
                img = Image.open(abs_filename).convert('L')
                ow,oh = img.size[0],img.size[1]
                imgmat = np.reshape(img.getdata(), [oh, ow])
                nw,nh=crop_image(imgmat,savepath=abs_filename)
                msg="Cropped image %s,%d-%d,%d-%d"%(fname,ow,oh,nw,nh)
                print("Cropped image %s,%d-%d,%d-%d"%(fname,ow,oh,nw,nh))
                f.write(msg+"\n")
    f.close()

def rescale_image(image_mat,nw=None,nh=None,fix_dim='H',normalize=False,invert=False,savepath=None):
    img=Image.fromarray(image_mat.astype('uint8')).convert('L')
    w,h=img.size[0],img.size[1]
    new_w=w
    new_h=h
    ar=w/float(h)
    if(fix_dim=='H'):
        new_w=int(ar*nh)
        new_h=nh
    elif(fix_dim=='W'):
        new_h=int(nw/float(ar))
        new_w=nw
    canvas = Image.new('L', (nw, new_h),color='white')
    rescaled=img.resize((new_w,new_h))
    canvas.paste(rescaled)
    if(invert):
        canvas=ImageOps.invert(canvas)
    new_image_mat=np.reshape(canvas.getdata(),[nh,nw])
    if(normalize):
        new_image_mat=new_image_mat/float(255.0)
    if(savepath is not None):
        img=Image.fromarray(new_image_mat).save(savepath)
    return new_image_mat,new_w,h

def batch_rescale_image(imagemats,nw,nh,fix_dim):
    rescale_image_mats=[]
    old_hs=[]
    old_ws=[]
    for mat in imagemats:
        new_mat,old_w,old_h=rescale_image(mat,nw=nw,nh=nh,fix_dim=fix_dim,normalize=False)
        #new_mat = rescale_image(mat, nw, nh, fix_dim, invert=False,normalize=False)
        #rescale_image_mats.append(new_mat)
        rescale_image_mats.append(new_mat)
        old_hs.append(old_h)
        old_ws.append(old_w)
    return rescale_image_mats,old_ws,old_hs

def test_image_rescaling(imfile,nw,nh,path):
    img=Image.open(imfile).convert('L')
    r,c=img.size[1],img.size[0]
    print(r,c)
    img=np.reshape(img.getdata(),(r,c))
    rescaled,nw,nh=rescale_image(img,nw,nh,savepath=path)
    ims=Image.fromarray(rescaled.astype('uint8')).convert('L').save('rescaled.png')
    print(rescaled.shape)

def list_to_string(lst,separator=" "):
    string=""
    for l in lst:
        string+=l+separator
    return string.rstrip(separator)

def read_config_file(configfile):
    config={}
    f=open(configfile)
    line=f.readline()
    while line:
        info=line.strip("\n").split(",")
        key=info[0]
        value=info[1]
        config[key]=value
        line=f.readline()
    print("Configuration loaded from %s"%configfile)
    return config

def bangla_to_unicode(banglastring):
    unicode_string=banglastring.encode('unicode-escape').decode('utf-8').replace("\\u"," \\u")
    return unicode_string

def decode_seq2seq_output(output_onehot,charmap,strip_meta=True):
    #output_onehot shape is dts,Nc
    this_label=""
    for t in range(len(output_onehot)):
        maxpos=np.argmax(output_onehot[t])
        char=charmap[maxpos]
        if(char!='<s>')and(char!='<e>')and(char!='<p>'):
            this_label=this_label+char
    return this_label
            
            