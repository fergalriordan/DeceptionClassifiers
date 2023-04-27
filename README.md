# DeceptionClassifiers
This repository contains the code used to apply a simple CART classifer and the Affinity Propagation algorithm to the five datasets used in the CODASPY 2022 poster “Does Deception Leave a Content Independent Stylistic Trace?”

# Datasets
The datasets used in this project can be found at: 
https://github.com/ReDASers/CODASPY-2022-Deception

# Files
t-SNE.py: code used to visualise the datasets

SimpleCART.py: implementation of a simple CART classifier

AffProp.py: implementation of the Affinity Propagation algorithm 

# Running the code
This code was developed on Google Colab, therefore the only instruction that can be given with confidence regarding the running of this code is to use the same platform. The t-SNE algorithm and the simple CART classifier should operate perfectly within this environment. The Affinity Propagation code may run out of RAM in the free version of Google Colab depending on the size of the dataset. If this occurs, the solution would be to upgrade to a paid version of Google Colab, to reduce the size of the dataset or to adapt the code for an alternative platform.

It is also important to note that the Affinity Propagation code in its present form does not converge for the deception datasets, and thus fails to produce a classifer. Thus, the implementation of the algorithm found in this repository is essentially useless: further research/experimentation is required to determine with confidence if this algorithm is suitable in this domain. 

# References/Credits
This code was produced by Fergal Riordan as a final project for the course 'Security Analytics' administered by Dr. Rakesh Verma at the University of Iceland in the spring semester of 2022/23. 
There were no other collaborators in the development of this code, though ChatGPT was used for some of its implementation. 
