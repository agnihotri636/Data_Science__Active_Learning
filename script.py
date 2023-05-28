#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import math
import warnings; warnings.simplefilter('ignore')


# # MINDREADING RANDOM SAMPLING

# In[2]:


#loop 
#logistic regression on train
#get accuracy
#select K=50 samples from unlabeled , pick their labels and add to training. remove them from unlabeled now
#repeat again 

print("Running random sampling for mindreading dataset......")
model=LogisticRegression(multi_class='multinomial')
score=pd.DataFrame(columns=['Accuracy_random'])
final_score=pd.DataFrame(columns=['Accuracy_random','Iteration_number'])
for i in range(1,4):
    training_matrix = scipy.io.loadmat('MindReading/trainingMatrix_MindReading'+str(i)+'.mat')
    training_labels= scipy.io.loadmat('MindReading/trainingLabels_MindReading_'+str(i)+'.mat')
    unlabeled_matrix= scipy.io.loadmat('MindReading/unlabeledMatrix_MindReading'+str(i)+'.mat')
    unlabeled_labels= scipy.io.loadmat('MindReading/unlabeledLabels_MindReading_'+str(i)+'.mat')
    testing_matrix= scipy.io.loadmat('MindReading/testingMatrix_MindReading'+str(i)+'.mat')
    testing_labels= scipy.io.loadmat('MindReading/testingLabels_MindReading'+str(i)+'.mat')
   
    training_matrix=pd.DataFrame(training_matrix["trainingMatrix"])
    training_labels=pd.DataFrame(training_labels["trainingLabels"])
    testing_matrix=pd.DataFrame(testing_matrix["testingMatrix"])
    testing_labels=pd.DataFrame(testing_labels["testingLabels"])
    unlabeled_matrix=pd.DataFrame(unlabeled_matrix["unlabeledMatrix"])
    unlabeled_labels=pd.DataFrame(unlabeled_labels["unlabeledLabels"])
    
        
    accuracy=[]
    for iteration in range(50): #K=50
           
        model.fit(training_matrix,training_labels)
        accuracy.append(model.score(testing_matrix,testing_labels))
        
        unlabeled_full=pd.concat([unlabeled_matrix, unlabeled_labels], axis=1)

        #N=10
        sample1=unlabeled_full.sample(10)

        sample1_label = pd.DataFrame(sample1.iloc[: , -1])
        sample1_matrix = pd.DataFrame(sample1.iloc[: , :-1])

        unlabeled_matrix = unlabeled_matrix.drop(sample1_matrix.index)
        unlabeled_labels = unlabeled_labels.drop(sample1_label.index)

        training_matrix=pd.concat([training_matrix,sample1_matrix])
        training_labels=pd.concat([training_labels,sample1_label])
        
# =============================================================================
#     print(training_matrix.shape +  training_labels.shape + testing_matrix.shape  + 
#           testing_labels.shape + unlabeled_matrix.shape + unlabeled_labels.shape)    
# =============================================================================
    score['Accuracy_random'] = accuracy
    score['Iteration_number'] = range(1,1+len(score))

    final_score=pd.concat([score,final_score])
  
    #plt.plot(score.Iteration_number,score.Accuracy)
    #plt.xlabel('Iteration_number')

    #plt.ylabel('Accuracy')
    #plt.show()

    
#final_score
#average of each run for mindreading dataset
final_score_MINDREAD_RANDOM=final_score.groupby(['Iteration_number']).mean().reset_index()
final_score

print("Finished run......")
# =============================================================================
# plt.plot(final_score_MINDREAD_RANDOM.Iteration_number, final_score_MINDREAD_RANDOM.Accuracy_random)
# plt.xlabel('Iteration_number')
# plt.ylabel('Accuracy_random')
# plt.title('Accuracy trend over iteration for Mindreading dataset using RANDOM SAMPLING')
# plt.show()
# 
# =============================================================================

# # MINDREADING UNCERATINITY BASED SAMPLING

# In[3]:


#loop 
#logistic regression on train
#get accuracy
#select K=50 samples from unlabeled , pick their labels and add to training. remove them from unlabeled now
#repeat again ?? how many times 

print("Running uncertainity based sampling for mindreading dataset......")
model=LogisticRegression(multi_class='multinomial')
score=pd.DataFrame(columns=['Accuracy_unceratin'])
final_score=pd.DataFrame(columns=['Iteration_number'])
for i in range(1,4):
    training_matrix = scipy.io.loadmat('MindReading/trainingMatrix_MindReading'+str(i)+'.mat')
    training_labels= scipy.io.loadmat('MindReading/trainingLabels_MindReading_'+str(i)+'.mat')
    unlabeled_matrix= scipy.io.loadmat('MindReading/unlabeledMatrix_MindReading'+str(i)+'.mat')
    unlabeled_labels= scipy.io.loadmat('MindReading/unlabeledLabels_MindReading_'+str(i)+'.mat')
    testing_matrix= scipy.io.loadmat('MindReading/testingMatrix_MindReading'+str(i)+'.mat')
    testing_labels= scipy.io.loadmat('MindReading/testingLabels_MindReading'+str(i)+'.mat')
   
    training_matrix=pd.DataFrame(training_matrix["trainingMatrix"])
    training_labels=pd.DataFrame(training_labels["trainingLabels"])
    testing_matrix=pd.DataFrame(testing_matrix["testingMatrix"])
    testing_labels=pd.DataFrame(testing_labels["testingLabels"])
    unlabeled_matrix=pd.DataFrame(unlabeled_matrix["unlabeledMatrix"])
    unlabeled_labels=pd.DataFrame(unlabeled_labels["unlabeledLabels"])
    
    
    
    accuracy=[]
    for iteration in range(50): #K=50
           
       
        model.fit(training_matrix,training_labels)
        accuracy.append(model.score(testing_matrix,testing_labels))
        
        probab=model.predict_proba(unlabeled_matrix)
        for i in range(len(probab)):
            for j in range(1,6):
                probab[i][j]=-math.log2(probab[i][j])*probab[i][j]
                
        final_entropy=[]       
        
        for i in range(len(probab)):
            total=0
            for j in range(0,6):
                total=total + probab[i][j]
            final_entropy.append(total)
            
        unlabeled_matrix["Entropy"]=final_entropy
             
        
        unlabeled_full=pd.concat([unlabeled_matrix, unlabeled_labels], axis=1)
        unlabeled_full=unlabeled_full.sort_values(['Entropy'], ascending=[0])
        
        #N=10 top 10 according to entropy
        sample1=unlabeled_full.head(10)
        sample1=sample1.drop(['Entropy'], axis=1)
        
        unlabeled_matrix=unlabeled_matrix.drop(['Entropy'], axis=1)
        
        sample1_label = pd.DataFrame(sample1.iloc[: , -1])
        sample1_matrix = pd.DataFrame(sample1.iloc[: , :-1])

        unlabeled_matrix = unlabeled_matrix.drop(sample1_matrix.index)
        unlabeled_labels = unlabeled_labels.drop(sample1_label.index)

        training_matrix=pd.concat([training_matrix,sample1_matrix])
        training_labels=pd.concat([training_labels,sample1_label])
        
# =============================================================================
#     print(training_matrix.shape +  training_labels.shape + testing_matrix.shape  + 
#           testing_labels.shape + unlabeled_matrix.shape + unlabeled_labels.shape)    
# =============================================================================
    score['Accuracy_uncertain'] = accuracy
    score['Iteration_number'] = range(1,1+len(score))

    final_score=pd.concat([score,final_score])
  
    #plt.plot(score.Iteration_number,score.Accuracy)
    #plt.xlabel('Iteration_number')

    #plt.ylabel('Accuracy')
    #plt.show()

    
#final_score
#average of each run for mindreading dataset
final_score_MINDREAD_UNCERTERTANITY=final_score.groupby(['Iteration_number']).mean().reset_index()
final_score

print("Finished run......")
# =============================================================================
# plt.plot(final_score_MINDREAD_UNCERTERTANITY.Iteration_number, final_score_MINDREAD_UNCERTERTANITY.Accuracy_uncertain)
# plt.xlabel('Iteration_number')
# plt.ylabel('Accuracy_uncertain')
# plt.title('Accuracy trend over iteration for MIND READING dataset using UNCERTAINITY BASED SAMPLING')
# plt.show()
# 
# =============================================================================

# # MMI RANDOM SAMPLING

# In[4]:


#Uncertainty-based Sampling: For each unlabeled sample, compute the classification entropy as
#e = - ∑ pi log (pi), where i runs from 1 to the number of classes and pi is the probability that the
#sample belongs to class i. Select the k samples producing the highest entropy. 


# In[5]:

#loop 
#logistic regression on train
#get accuracy
#select K=50 samples from unlabeled , pick their labels and add to training. remove them from unlabeled now
#repeat again 

print("Running random sampling for MMI dataset......")
model=LogisticRegression(multi_class='multinomial')
score=pd.DataFrame(columns=['Accuracy_random'])
final_score=pd.DataFrame(columns=['Accuracy_random','Iteration_number'])
for i in range(1,4):
    training_matrix = scipy.io.loadmat('MMI/trainingMatrix_'+str(i)+'.mat')
    training_labels= scipy.io.loadmat('MMI/trainingLabels_'+str(i)+'.mat')
    unlabeled_matrix= scipy.io.loadmat('MMI/unlabeledMatrix_'+str(i)+'.mat')
    unlabeled_labels= scipy.io.loadmat('MMI/unlabeledLabels_'+str(i)+'.mat')
    testing_matrix= scipy.io.loadmat('MMI/testingMatrix_'+str(i)+'.mat')
    testing_labels= scipy.io.loadmat('MMI/testingLabels_'+str(i)+'.mat')
   
    training_matrix=pd.DataFrame(training_matrix["trainingMatrix"])
    training_labels=pd.DataFrame(training_labels["trainingLabels"])
    testing_matrix=pd.DataFrame(testing_matrix["testingMatrix"])
    testing_labels=pd.DataFrame(testing_labels["testingLabels"])
    unlabeled_matrix=pd.DataFrame(unlabeled_matrix["unlabeledMatrix"])
    unlabeled_labels=pd.DataFrame(unlabeled_labels["unlabeledLabels"])
    
    
    
    accuracy=[]
    for iteration in range(50): #K=50
           
        
        model.fit(training_matrix,training_labels)
        accuracy.append(model.score(testing_matrix,testing_labels))
        
        unlabeled_full=pd.concat([unlabeled_matrix, unlabeled_labels], axis=1)

        #N=10
        sample1=unlabeled_full.sample(10)

        sample1_label = pd.DataFrame(sample1.iloc[: , -1])
        sample1_matrix = pd.DataFrame(sample1.iloc[: , :-1])

        unlabeled_matrix = unlabeled_matrix.drop(sample1_matrix.index)
        unlabeled_labels = unlabeled_labels.drop(sample1_label.index)

        training_matrix=pd.concat([training_matrix,sample1_matrix])
        training_labels=pd.concat([training_labels,sample1_label])
        
# =============================================================================
#     print(training_matrix.shape +  training_labels.shape + testing_matrix.shape  + 
#           testing_labels.shape + unlabeled_matrix.shape + unlabeled_labels.shape)    
# =============================================================================
    score['Accuracy_random'] = accuracy
    score['Iteration_number'] = range(1,1+len(score))

    final_score=pd.concat([score,final_score])
  
    #plt.plot(score.Iteration_number,score.Accuracy)
    #plt.xlabel('Iteration_number')

    #plt.ylabel('Accuracy')
    #plt.show()

    
#final_score
#average of each run for mindreading dataset
final_score_MMI_RANDOM=final_score.groupby(['Iteration_number']).mean().reset_index()
final_score

print("Finished run......")
# =============================================================================
# plt.plot(final_score_MMI_RANDOM.Iteration_number, final_score_MMI_RANDOM.Accuracy_random)
# plt.xlabel('Iteration_number')
# plt.ylabel('Accuracy_random')
# plt.title('Accuracy trend over iteration for MMI dataset using RANDOM SAMPLING')
# plt.show()
# =============================================================================


# # MMI UNCERTAINITY BASED SAMPLING
# 

# In[6]:


#loop 
#logistic regression on train
#get accuracy
#select K=50 samples from unlabeled , pick their labels and add to training. remove them from unlabeled now
#repeat again ?? how many times 

print("Running uncertainity based sampling for mindreading dataset......")
model=LogisticRegression(multi_class='multinomial')
score=pd.DataFrame(columns=['Accuracy_unceratin'])
final_score=pd.DataFrame(columns=['Iteration_number'])
for i in range(1,4):
    training_matrix = scipy.io.loadmat('MMI/trainingMatrix_'+str(i)+'.mat')
    training_labels= scipy.io.loadmat('MMI/trainingLabels_'+str(i)+'.mat')
    unlabeled_matrix= scipy.io.loadmat('MMI/unlabeledMatrix_'+str(i)+'.mat')
    unlabeled_labels= scipy.io.loadmat('MMI/unlabeledLabels_'+str(i)+'.mat')
    testing_matrix= scipy.io.loadmat('MMI/testingMatrix_'+str(i)+'.mat')
    testing_labels= scipy.io.loadmat('MMI/testingLabels_'+str(i)+'.mat')
   
    training_matrix=pd.DataFrame(training_matrix["trainingMatrix"])
    training_labels=pd.DataFrame(training_labels["trainingLabels"])
    testing_matrix=pd.DataFrame(testing_matrix["testingMatrix"])
    testing_labels=pd.DataFrame(testing_labels["testingLabels"])
    unlabeled_matrix=pd.DataFrame(unlabeled_matrix["unlabeledMatrix"])
    unlabeled_labels=pd.DataFrame(unlabeled_labels["unlabeledLabels"])
    
    
   
    accuracy=[]
    for iteration in range(50): #K=50
           
       
        model.fit(training_matrix,training_labels)
        accuracy.append(model.score(testing_matrix,testing_labels))
        
        probab=model.predict_proba(unlabeled_matrix)
        for i in range(len(probab)):
            for j in range(1,6):
                probab[i][j]=-math.log2(probab[i][j])*probab[i][j]
                
        final_entropy=[]       
        
        for i in range(len(probab)):
            total=0
            for j in range(0,6):
                total=total + probab[i][j]
            final_entropy.append(total)
            
        unlabeled_matrix["Entropy"]=final_entropy
             
        
        unlabeled_full=pd.concat([unlabeled_matrix, unlabeled_labels], axis=1)
        unlabeled_full=unlabeled_full.sort_values(['Entropy'], ascending=[0])
        
        #N=10 top 10 according to entropy
        sample1=unlabeled_full.head(10)
        sample1=sample1.drop(['Entropy'], axis=1)
        
        unlabeled_matrix=unlabeled_matrix.drop(['Entropy'], axis=1)
        
        sample1_label = pd.DataFrame(sample1.iloc[: , -1])
        sample1_matrix = pd.DataFrame(sample1.iloc[: , :-1])

        unlabeled_matrix = unlabeled_matrix.drop(sample1_matrix.index)
        unlabeled_labels = unlabeled_labels.drop(sample1_label.index)

        training_matrix=pd.concat([training_matrix,sample1_matrix])
        training_labels=pd.concat([training_labels,sample1_label])
        
# =============================================================================
#     print(training_matrix.shape +  training_labels.shape + testing_matrix.shape  + 
#           testing_labels.shape + unlabeled_matrix.shape + unlabeled_labels.shape)    
# =============================================================================
    score['Accuracy_uncertain'] = accuracy
    score['Iteration_number'] = range(1,1+len(score))

    final_score=pd.concat([score,final_score])
  
    #plt.plot(score.Iteration_number,score.Accuracy)
    #plt.xlabel('Iteration_number')

    #plt.ylabel('Accuracy')
    #plt.show()

    
#final_score
#average of each run for mindreading dataset
final_score_MMI_UNCERTERTANITY=final_score.groupby(['Iteration_number']).mean().reset_index()
final_score

print("Finished run......")
# =============================================================================
# plt.plot(final_score_MMI_UNCERTERTANITY.Iteration_number, final_score_MMI_UNCERTERTANITY.Accuracy_uncertain)
# plt.xlabel('Iteration_number')
# plt.ylabel('Accuracy_uncertain')
# plt.title('Accuracy trend over iteration for MIND READING dataset using UNCERTAINITY BASED SAMPLING')
# plt.show()
# 
# =============================================================================

# ### COMPARING RESULTS FOR BOTH TECHNIQUES FOR MIND READING DATASET

# In[7]:

print("Plotting iteration vs accuracy curves......")    

final_score_Mindreading=pd.concat([final_score_MINDREAD_RANDOM, final_score_MINDREAD_UNCERTERTANITY], axis=1)
final_score_Mindreading
final_score_Mindreading = final_score_Mindreading.loc[:,~final_score_Mindreading.columns.duplicated()].copy()

plt.plot(final_score_Mindreading.Iteration_number, final_score_Mindreading.Accuracy_uncertain, "-b", label="UNCERTAINITY BASED SAMPLING")
plt.plot(final_score_Mindreading.Iteration_number, final_score_Mindreading.Accuracy_random, "-r", label="RANDOM SAMPLING")
plt.legend(loc="upper left")
plt.xlabel('Iteration_number')
plt.ylabel('Accuracy')
plt.title("MIND READING")

plt.show()


# ### COMPARING RESULTS FOR BOTH TECHNIQUES FOR MMI DATASET

# In[8]:

final_score_MMI=pd.concat([final_score_MMI_RANDOM, final_score_MMI_UNCERTERTANITY], axis=1)
final_score_MMI
final_score_MMI = final_score_MMI.loc[:,~final_score_MMI.columns.duplicated()].copy()

plt.plot(final_score_MMI.Iteration_number, final_score_MMI.Accuracy_uncertain, "-b", label="UNCERTAINITY BASED SAMPLING")
plt.plot(final_score_MMI.Iteration_number, final_score_MMI.Accuracy_random, "-r", label="RANDOM SAMPLING")
plt.legend(loc="upper left")
plt.xlabel('Iteration_number')
plt.ylabel('Accuracy')
plt.title("MMI")
plt.show()

