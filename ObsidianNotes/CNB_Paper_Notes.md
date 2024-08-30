sqTo read:
- Skim Section 2.1: PPI Network
- Important section 3.2: network paths and distances
- 4.5 Machine learning in biology
- 4.5.1 focus on the math part
- Skim through some of the section 5 applications

### 2.1 Notes:
Protein-protein interactions (PPI) are physical links/interactions between two or more proteins. Specifically though direct contact. This paper looked at the strategy for mapping Human PPIs through a process called yeast two-hybrid assay (Y2H) (don't need to know what it means).

The PPIs studied could be grouped into six different experimental types.
1. Binary Y2H
	1. Do two proteins interact, yes or no
	2. Very very cheap to run
	3. Uses yeast mating
	4. Low confidence
2. 3-dimensional structure-solved
	1. We have the 3D structure of proteins
	2. Can identify visually whether or not a protein could interact
	3. very expensive, and high confidence
3. Kinase-substrate interactions (from databases)
	1. Directed PPI
	2. Modifications to protein as interactions
4. Signaling networks (from databases)
	1. Like from earlier
5. protein complexes
	1. When proteins form a complex. Like if proteins A,B,C merge together
6. carefully literature-curated PPIs
	2. Stated through multiple papers, automatically added to databases

SPRAS will likely pull data from Kinase-substrate databases and signaling network databases

### 3.2 Notes: Network Path and Distance
Oftentimes, when talking about networks, one must discuss the **shortest path**. The shorted path in this context is the network path with the shortest lengths between node pairs (the document uses circles and triangles to symbolize this). More technically:

For the node pair $(i,j)\in V$, the shortest path length is defined as $d_{i,j}$ for the node pair.

Guney et al. proposes a bunch of measurements to apply to a graph:
* closest = $d_{c}$
* shortest = $d_s$
* kernel = $d_k$
* center = $d_{cc}$
* separation = $d_{ss}$

These are used primarily to evaluate the distance between drugs and diseases (where drugs and diseases are represented as a set of drug target proteins \[ask anna]). These distances can be calculated (equations are shown on page 12 of the paper).

#### Network Efficiency:
Based on shortest bath, there is a metric called network efficiency which is calculated as: $$NE = \frac{1}{N(N-1)} \sum_{{i\neq j}} \frac{1}{d_{{i,j}}}$$
The network efficiency is a measure of the "traffic capacity" of a network. Network efficiency is widely used in brain functional networks (more effieicnt parallel information transfer).

#### Betweenness and bottlenecks:
Betweenness is a path-based measure that measures importance. Specifically in how many shortest paths go through a certain node. Betweenness is calculated as: $$B_{v} = \sum_{i,j,v\in V, neq j} \frac{\delta_{i,j}(v)}{\delta_{i,j}}$$
The betweenness of a node follows power-law distribution in Cayley trees (look up later). Note, the correlation between $B_v$ and the degree of a node is not always positive. Some nodes may have a small degree but a very large betweenness, which serves as important connections between modules.

Nodes with high betweenness can also serve as bottlenecks for a network. Bottlenecks are nodes that control most of the information flow in a network, and if they are removed, the network efficiency would decrease significantly.

There are a couple more metrics relating to betweenness and bottlenecking that are important to note:

**Double Specific Betweenness (S2B)** (Garcia-Vaquero et al.), to measure a node's information traffic between nodes in different sets. $$S2B_{v}(s_{1},s_{2})=\sum_{i\in s_{1},j\in s_{2},i\neq j,v\in V} \frac{\delta_{i,j}(v)}{\delta_{i,j}}$$
**Edge Betweenness ($B_e$)**, to measure the betweenness of edges. It is calculated as: $$B_{e} = \sum_{i,j\in V,i\neq j, e\in E} \frac{\delta_{i,j}(e)}{\delta_{i,j}}$$
### 4.5 Notes: Machine Learning in Network Biology
Machine-learning is the backbone of artificial intelligence and data science. The methods in machine-learning operate through generating a predictive model. In order to do so, the model must be fed multiple types of high-quality data. This is accomplished through the high-throughput technology that has emerged recently.

When using machine-learning, we want to prepare the data so features are easily available for use. Some features are:
##### - Network Based
- Node centrality
- node interaction
- local structure
- subgraphs
- network propagation results
- network-based similarities
##### - Biological Information
- gene expression profile
- gene mutation frequency
- gene functional annotation

The learning in machine learning works to try and find a model that will map input features into accurate predictions. There are two methods of learning, supervised and unsupervised. Supervised provides the labels for the model to map to, essentially a cheat-sheet.

#### Important ML Math stuff!!!
There are a bunch of equations that are used when figuring out how effective a model is. In supervised learning, the input space (features) is classified as $x = (x_{0},x_{1},\dots,x_{m})\in X$, and the output space (label) are classified as $y\in Y$. The ultimate goal is to choose a function $f_{\omega}(x)$ with a parameter vector $\omega$ that will predict the label $y$ the best. We can quantify the "goodness"  of a function by finding the minimum of the follow loss function:$$L(\omega)=\sum_{x\in X,y\in Y}\Theta(y,f_{\omega}(x))$$ where $\Theta(*,*)$ is the gap between the label and the prediction. \[There is a bunch of fancy math I am gonna ask Anna to walk through]


