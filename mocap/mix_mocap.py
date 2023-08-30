import numpy as np
import os

two_train=np.load('two_train_4seconds_2.npy',allow_pickle=True)
one_train=np.load('one_train_4seconds_30.npy',allow_pickle=True)

print(two_train.shape)
print(one_train.shape)


# 3000 sequences have 2 subjects and 1 single subject, 3000 sequences have 3 single subject

two_sample=np.random.choice(len(two_train),3000)
one_sample=np.random.choice(len(one_train),3000+3000*2)

one=one_sample[:3000] #mix with two

one_1=one_sample[3000:6000]
one_2=one_sample[6000:9000]

data=[]
for i in range(6000):
    #3000 sequences have 2 subjects and 1 single subject
    if i<3000:
        two_person=two_train[two_sample[i]]

        #random initialization
        two_person[:,:,:,[0,2]]=two_person[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
        temp=np.concatenate([two_person]) 
        #put the whole scene into the center
        temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0])
        temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) 
        temp=temp.reshape(2,120,-1) 
        data.append(temp)

    #3000 sequences have 3 single subject
    # else:
    #     one_person_1=one_train[one_1[i-3000]]
    #     one_person_2=one_train[one_2[i-3000]]
    #     one_person_1[:,:,:,[0,2]]=one_person_1[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    #     one_person_2[:,:,:,[0,2]]=one_person_2[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    #     temp=np.concatenate([one_person_1,one_person_2]) 
    #     temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0])
    #     temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) 
    #     temp=temp.reshape(2,120,-1)
    #     data.append(temp)

data=np.array(data) # 6000 sequences, 3 persons, 120 (30 fps 4 seconds), 93 joints xyz (31x3)
print(data.shape)

# default joints from CMU Mocap
# use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27] #used joints and order

# our joints
use = [14,17,24,19,26,20,27,21,28]
data=data.reshape(data.shape[0],2,-1,31,3)
data=data[:,:,:,use,:]
data=data.reshape(data.shape[0],2,-1,len(use)*3)
#In order to mix the data from different sources, we scale different data respectively in this code. 
#This may make the result slightly different from the table in the paper.
data=data*(1.0/0.45)*2.54/100.0
np.save('train_3_120_mocap.npy',data)



###########################################################################

#test data

two_test=np.load('two_test_4seconds_2.npy',allow_pickle=True)
one_test=np.load('one_test_4seconds_30.npy',allow_pickle=True)

print(two_test.shape)
print(one_test.shape)

#400 sequences have 2 subjects and 1 single subject
#400 sequences have 3 single subject

two_sample=np.random.choice(len(two_test),400)
one_sample=np.random.choice(len(one_test),400+400*2)

one_1=one_sample[400:800]
one_2=one_sample[800:1200]

data=[]
for i in range(800):
    #800 sequences have 2 subjects and 1 single subject
    if i<400:
        two_person=two_test[two_sample[i]]
        two_person[:,:,:,[0,2]]=two_person[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
        temp=np.concatenate([two_person]) 
        temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0])
        temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) 
        temp=temp.reshape(2,120,-1) 
        data.append(temp)


    # else:
    #     one_person_1=one_test[one_1[i-400]]
    #     one_person_2=one_test[one_2[i-400]]
    #     one_person_1[:,:,:,[0,2]]=one_person_1[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    #     one_person_2[:,:,:,[0,2]]=one_person_2[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    #     temp=np.concatenate([one_person_1,one_person_2]) 
    #     temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0])
    #     temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) 
    #     temp=temp.reshape(2,120,-1)
    #     data.append(temp)

data=np.array(data)

# default joints from CMU Mocap
# use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27] #used joints and order

# our joints
use = [14,17,24,19,26,20,27,21,28]
data=data.reshape(data.shape[0],2,-1,31,3)
data=data[:,:,:,use,:]
data=data.reshape(data.shape[0],2,-1,len(use)*3)
data=data*(1.0/0.45)*2.54/100.0 # scale
print(data.shape)
np.save('test_3_120_mocap.npy',data)



###########################################################################

#discriminator data
one_train=np.load('one_train_4seconds_30.npy',allow_pickle=True)
print(one_train.shape)

# 6000 have 3 single subject

one_sample=np.random.choice(len(one_train),6000*2)



one_1=one_sample[:6000]
one_2=one_sample[6000:12000]

data=[]
for i in range(6000):
    
    one_person_1=one_train[one_1[i]]
    one_person_2=one_train[one_2[i]]
    one_person_1[:,:,:,[0,2]]=one_person_1[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    one_person_2[:,:,:,[0,2]]=one_person_2[:,:,:,[0,2]]+np.array([np.random.randint(-50,50),np.random.randint(-50,50)])
    temp=np.concatenate([one_person_1,one_person_2]) 
    temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0])
    temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) 
    temp=temp.reshape(2,120,-1)
    data.append(temp)

data=np.array(data) 

# default joints from CMU Mocap
# use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27] #used joints and order

# our joints
use = [14,17,24,19,26,20,27,21,28]
data=data.reshape(data.shape[0],2,-1,31,3)
data=data[:,:,:,use,:]
data=data.reshape(data.shape[0],2,-1,len(use)*3)
data=data*(1.0/0.45)*2.54/100.0 # scale
print(data.shape)
np.save('discriminator_3_120_mocap.npy',data)
