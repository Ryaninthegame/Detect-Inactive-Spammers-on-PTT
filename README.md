# Using-Graph-Neural-Networks-to-Detect-Inactive-Spammers-on-PTT
## Set Up
* Python : 3.6.10
* Pytorch : 1.6.0
* DGL : 0.6.1
* scipy : 1.5.2
* sklearn : 0.23.2
* numpy : 1.19.1
* argparse : 1.1

## Quickstart
```
unzip ./data/userSet.zip
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
      $ python GCN_train.py  
            --withSuspectValue <Whether to use the suspect value (default:True)> 
            --run <Decide how many times to run (default:10)> 
      ```
    * Example:
      ```
      $ python GCN_train.py  
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
            --epoch <epoch (default:1000)>
            --K <TAGCN Hyperparameter (default:3)>
      ```
    * Example:
      ```
      $ python TAGCN_train.py  
            --withSuspectValue True
            --run 10
            --cutting 3
            --epoch 1000
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
