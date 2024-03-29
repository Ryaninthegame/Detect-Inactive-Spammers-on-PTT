# Detect-Inactive-Spammers-on-PTT

This repo contains the implementation for the paper:
**Detecting Inactive Cyberwarriors from Online Forums**, published at IEEE International Conference on Web Intelligence and Intelligent Agent Technology (WI) 2023.

This study investigates the activity of cyberwarriors on a large online forum, revealing that only a small portion of them are active users. Interestingly, despite their role in spreading misinformation, these cyberwarriors typically remain silent during peaceful periods, becoming active only when needed. The research explores the difficulties in identifying cyberwarriors, particularly inactive ones, which are more challenging than detecting their active counterparts. The study suggests utilizing social information to detect cyberwarriors more effectively during their inactivity periods.

## Setup
Tested under Python 3.8.13 in Ubuntu.

Install the required packages by
```
$ pip install -r requirements.txt
```

## File Description
The following files are under folder data/information/
* adjacentMatrix.npz : Adjacent matrix, shape:(44602, 44602)
* activeValue.json : User's active value
* suspectValue,json : User's suspect value
* indexUser.json : User -> adjacent matrix index
* userIndex.json : Adjacent matrix index -> user
* normalUser.json : Normal user list
* spammer.json : Spammer list
* userLabel.json : User label, normal user is 0, spammer is 1

## Quickstart
```
$ unzip ./data/userSet.zip
```
### GCN
* Train
    * Usage :  
      ```
      $ python GCN_train.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
            --cutting < Batch = (DataSize/cutting), Smaller cutting, larger Batch size (default:3)>
            --epoch <epoch (default:1000)>
      ```
    * Example:
      ```
      $ python GCN_train.py  
            --withSuspectValue True
            --run 10
            --cutting 3
            --epoch 1000
      ```
* Test
    * Usage :  
      ```
      $ python GCN_test.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
      ```
    * Example:
      ```
      $ python GCN_test.py  
            --withSuspectValue True
            --run 10
      ```

### TAGCN
* Train
    * Usage :  
      ```
      $ python TAGCN_train.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
            --cutting < Batch = (DataSize/cutting), Smaller cutting, larger Batch size (default:3)>
            --epoch <epoch (default:5000)>
            --K <TAGCN Hyperparameter (default:3)>
      ```
    * Example:
      ```
      $ python TAGCN_train.py  
            --withSuspectValue True
            --run 10
            --cutting 3
            --epoch 5000
            --K 3
      ```
* Test
    * Usage :  
      ```
      $ python TAGCN_test.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
      ```
    * Example:
      ```
      $ python TAGCN_test.py  
            --withSuspectValue True
            --run 10
      ```
      
### GAT
* Train
    * Usage :  
      ```
      $ python GAT_train.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
            --cutting < Batch = (DataSize/cutting), Smaller cutting, larger Batch size (default:3)>
            --epoch <epoch (default:1000)>
            --numHeads <GAT Hyperparameter (default:4)>
      ```
    * Example:
      ```
      $ python GAT_train.py  
            --withSuspectValue True
            --run 10
            --cutting 3
            --epoch 1000
            --numHeads 4
      ```
* Test
    * Usage :  
      ```
      $ python GAT_test.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
      ```
    * Example:
      ```
      $ python GAT_test.py  
            --withSuspectValue True
            --run 10
      ```

## Citation
Please cite the paper if you find the work useful.

    @inproceedings{wang23detecting,
       title={Detecting Inactive Cyberwarriors from Online Forums},
       author={Ruei-Yuan Wang and Hung-Hsuan Chen},
       booktitle={International Conference on Web Intelligence and Intelligent Agent Technology},
       year={2023}
     }
