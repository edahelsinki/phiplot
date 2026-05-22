<div style="display: flex; gap: 1rem; max-width: 1000px; margin: 0 auto;">

<nav style="max-width: 350px; max-height: 100vh; position: sticky; top: 0; overflow-y: auto; border-left: 2px solid gray; padding: 20px">

## Table of Contents
- [1. Molecular Fingerprinting](#molecular-fingerprinting)
    - [1.1 Morgan](#morgan)
    - [1.2 RDKit Fingerprint](#rdkit-fingerprint)
    - [1.3 TopologicalTorsions](#topologicaltorsions)
    - [1.4 MACCS](#maccs-key)
    - [1.5 ATMOMACCS](#atmomaccs)
- [2. Clustering Analysis](#clustering-analysis)
    - [2.2 Clustering Metrics](#clustering-metrics)
    - [2.2 K-means](#k-means)
    - [2.3 HAC](#hac)
    - [2.3 BIRCH](#birch)
    - [2.4 Bisecting K-means](#bisecting-k-means)
- [3. Kernel Methods](#3-kernel-methods)
    - [3.1 Linear Kernel](#31-linear-kernel)
    - [3.2 Polynomial Kernel](#32-polynomial-kernel)
    - [3.3 RBF Kernel](#33-rbf-kernel)
    - [3.4 Cosine Kernel](#34-cosine-kernel)
    - [3.5 Manhattan Kernel](#35-manhattan_kernel)
    - [3.6 Hamming Kernel](#36-hamming_kernel)
    - [3.7 Tanimoto Kernel](#37-tanimoto-kernel)
- [4. Unsupervised Embedding](#4-unsupervised-embedding)
    - [4.1 PCA](#41-pca)
    - [4.2 KPCA](#42-kpc)
    - [4.3 tSNE](#43-tsne)
- [5. Semi-Supervised Embedding](#5-semi-supervised-embedding)
    - [5.1 LSP](#51-lsp)
    - [5.2 cKPCA](#52-ckpca)

</nav>

<main style="flex: 1; border-left: 2px solid gray; border-right: 2px solid gray; padding: 20px">

<h1 style="display: flex; align-items: center; gap: 15px;">
    <img src="app/assets/figures/logo.png" alt="logo" width="50px"/>
    PhiPlot Documentation
</h1>

---

# <a id="molecular-fingerprinting"></a> 1. Molecular Fingerprinting

### Molecular Graphs
At the fundamental level, molecular structures are represented as 2D molecular graphs where vertices correspond to atoms and edges represent chemical bonds. Hydrogen atoms and their bonds are typically omitted (implicit hydrogens). 

To be chemically meaningful, these graphs must satisfy three key invariances:
* **Rotational and Translational Invariance:** The identity of the molecule remains constant regardless of its position or orientation in space.
* **Permutation Invariance:** The molecule remains identical regardless of the indexing order of its atoms.

### Linear Notations
Cheminformatics relies on alphanumeric notations, the most prevalent being **SMILES** (Simplified Molecular Input Line Entry System).
* **Atoms:** Represented by atomic symbols (upper-case for aliphatic, lower-case for aromatic).
* **Bonds:** Single bonds are implicit; double and triple bonds use `=` and `#`.
* **Topology:** Branches are enclosed in parentheses `()`; rings are indicated by matching numerical labels.

> **Note on Canonicalisation:** Because a single molecule can have multiple valid SMILES strings, canonicalisation algorithms (like Canonical SMILES or IUPAC’s InChI) are used to ensure a unique, one-to-one mapping.

### Fingerprinting
To utilise chemical structures in machine learning, they must be transformed into fixed-length numerical vectors. While **molecular descriptors** refer to any physical property, **fingerprints** specifically refer to vectors (often binary or integer-valued) encoding the presence or frequency of substructures.

Fingerprinting algorithms are categorised by how they traverse the molecular graph:

| Category | Methodology | Characteristics |
| :--- | :--- | :--- |
| **Circular** | Iteratively aggregates info from neighbouring atoms in increasing radii. | Uses hashing; can represent an almost infinite variety of substructures. |
| **Structural Keys** | Relies on a predefined "dictionary" of specific fragments (e.g., carboxyl groups). | Each bit corresponds to a specific, human-defined substructure. |
| **Path-Based** | Identifies all linear paths of a specific bond length (usually 1–7 bonds). | Highly effective for substructure searching. |
| **Linear/Torsion** | Focuses on sequences of four bonded atoms. | Captures the "skeleton" or shape rather than local clusters. |

#### Implementation via RDKit
The **RDKit** Python library is the primary tool used in this application to generate these fingerprints and handle general cheminformatics logic.

## <a id="morgan"></a> 1.1 Morgan

---

The Morgan fingerprint is RDKit's implementation of the extended connectivity fingerprint (ECFP). It is a circular fingerprint that considers the environment of each atom within an increasing radius. It excels at capturing local chemical environments and is the gold standard for modelling most molecular properties and bioactivity. For each atom, the algorithm first identifies its immediate neighbours, then the neighbours of those neighbours, and so on up to a user-defined radius. At each step, the atom's environment is converted into an integer.

| Parameter | Description | Type | Possible Values | Default |
| :--- | :--- | :--- | :--- | :--- |
| **`radius`** | Number of iterations to grow circular environments. | `int` | `>=0` | `2` |
| **`fpSize`** | Total number of bits in the bit vector. | `int` | `>=1` | `2048` |
| **`includeChirality`** | Includes stereochemistry in atom invariants. | `bool` | `true`, `false` | `false` |
| **`useBondTypes`** | Includes bond orders in the invariants. | `bool` | `true`, `false` | `true` |
| **`useCounts`** | Produces a count-based vector if enabled. | `bool` | `true`, `false` | `false` |
| **`onlyNonzeroInvariants`** | Uses only non-zero atom invariants for hashing. | `bool` | `true`, `false` | `false` |
| **`includeRingMembership`** | Adds ring membership info to invariants. | `bool` | `true`, `false` | `true` |
| **`includeRedundantEnvironments`** | Retains overlapping environments. | `bool` | `true`, `false` | `false` |
| **`useFeatures`** | Uses functional group features (FCFP-style). | `bool` | `true`, `false` | `false` |
| **`countSimulation`** | Simulates counts within a bit vector. | `bool` | `true`, `false` | `false` |

## <a id="rdkit-fingerprint"></a> 1.2 RDKit Fingerprint

---

This is a path-based fingerprint that is unique to the RDKit library. It is specifically designed for use in substructure searching and screening. As every possible subpath of a molecule is encoded, if molecule A is a substructure of molecule B, all the bits set for A will also be set for B (except in the case of rare hash collisions). The algorithm identifies all linear paths (and optionally branched subgraphs) within a specific bond length range. By default, it considers paths ranging from 1 to 7 bonds. Each unique path is hashed and this hash is used as a seed for a pseudorandom number generator that sets a specific number of bits in the fingerprint.

| Parameter | Description | Type | Possible Values | Default |
| :--- | :--- | :--- | :--- | :--- |
| **`minPath`** | Minimum number of bonds in a path. | `int` | `>=1` | `1` |
| **`maxPath`** | Maximum number of bonds in a path. | `int` | `>=1` | `7` |
| **`useHs`** | Includes explicit hydrogens in path identification. | `bool` | `true`, `false` | `true` |
| **`branchedPaths`** | Considers branched subgraphs if enabled. | `bool` | `true`, `false` | `true` |
| **`useBondOrder`** | Distinguishes paths by specific bond orders. | `bool` | `true`, `false` | `true` |
| **`countSimulation`** | Simulates path frequency in the bit vector. | `bool` | `true`, `false` | `false` |
| **`fpSize`** | Total bits available for hashing paths. | `int` | `>=1` | `2048` |
| **`numBitsPerFeature`** | Number of bits set per unique feature found. | `int` | `>=1` | `2` |

## <a id="topologicaltorsions"></a> 1.3 TopologicalTorsions

---

The TopologicalTorsions torsion descriptor type fingerprint represents the 'short-range' skeletal information of a molecule, capturing the sequence of four bonded atoms. It is highly effective at capturing the specific connectivity and 'topology' of the molecular backbone, offering an alternative perspective to the 'cluster-based' view of Morgan fingerprints. The algorithm identifies all paths consisting of exactly four non-hydrogen atoms (A–B–C–D). For each path, it calculates a descriptor based on the types of the atoms (i.e. their atomic number), the number of non-hydrogen neighbours of each atom, and the number of π electrons of each atom.

| Parameter | Description | Type | Possible Values | Default |
| :--- | :--- | :--- | :--- | :--- |
| **`includeChirality`** | Includes stereochemistry in torsion sequences. | `bool` | `true`, `false` | `false` |
| **`torsionAtomCount`** | Number of atoms in the torsion path. | `int` | `>=2` | `4` |
| **`countSimulation`** | Simulates torsion counts via bit-masking. | `bool` | `true`, `false` | `true` |
| **`fpSize`** | Total size of the hashed torsion bit vector. | `int` | `>=1` | `2048` |
| **`ownsAtomInvGen`** | Internal memory management flag. | `bool` | `true`, `false` | `false` |

## <a id="maccs-key"></a> 1.4 MACCS Key

---

The MACCS key fingerprint, which is based on structural keys, is RDKit’s implementation of the 166 public structural keys originally developed by MDL Information Systems. It is a well-established standard in the field and is particularly valued for its interpretability and efficiency in substructure screening and rapid similarity searching. For each molecule, the algorithm checks the structure against a fixed list of predefined queries, setting the corresponding bit to 1 if a feature is found.

| Parameter | Description | Type | Possible Values | Default |
| :--- | :--- | :--- | :--- | :--- |
| **`useCounts`** | Returns count-based vector instead of binary. | `bool` | `true`, `false` | `false` |

## <a id="atmomaccs"></a> 1.5 ATMOMACCS

---

ATMOMACCS is an interpretable, hybrid molecular descriptor specifically designed for machine learning applications in atmospheric science. Developed by researchers at Aalto University and the Technical University of Munich, it addresses the limitations of general-purpose fingerprints in capturing the complex, highly oxidized organic compounds common in the atmosphere.

ATMOMACCS combines the structural connectivity information of the standard MACCS keys with atmospheric domain knowledge derived from the SIMPOL group contribution method. While traditional fingerprints often overlook structural characteristics relevant to atmospheric chemistry, such as specific oxygen- and nitrogen-rich functional groups, ATMOMACCS explicitly incorporates these "ATMO" motifs to improve predictive accuracy and model interpretability.

| Parameter | Description | Type | Possible Values | Default |
| :--- | :--- | :--- | :--- | :--- |
| **`version`** | Architecture version for the ATMO component. | `int` | `1, 2, 3, 4` | `4` |
| **`bit_width`** | Bit width for binary encoding of C/O counts. | `int` | `>=1` | `6` |

<br>

| Version | Keys | Encoding | Key Differences |
| :--- | :--- | :--- | :--- |
| **`V1`** | 202 | Binary | Presence/absence of SIMPOL groups. |
| **`V2`** | 274 | Binary | Presence of SIMPOL groups in 0, 1, 2, or > 2 instances. |
| **`V3`** | 280 | Binary | V2 plus binary encoding of carbon atom count. |
| **`V4`** | 286 | Binary | V3 plus binary encoding of oxygen atom count. |
| **`V5`** | 204 | Integer | Absolute counts of all MACCS and ATMO keys. |


# <a id="clustering-analysis"></a> 2. Clustering Analysis

---

Clustering analysis is a fundamental component of **unsupervised machine learning**, used to analyse data sets without labelled target variables. It enables the discovery of hidden patterns and structures by partitioning data points into distinct clusters based on inherent similarities.

### Core Principles
The goal is to ensure data points within the same cluster exhibit greater similarity to each other than to those in different clusters. This documentation covers two primary classes:
* **Centroid Models:** Organise data points based on proximity to representative central points.
* **Connectivity Models:** Identify hierarchical, tree-like relationships.
* **Hybrid Models:** Combine features from both approaches.

### Mathematical Framework
We denote the set of $N$ observations as $$X = \{\mathbf{x}\_i \in \mathbb{R}^p\}_{i=1}^N$$, where each element is a $$p$$-dimensional real vector. The set of $k$ clusters is $$S = \{S\_i\}\_{i=1}^k$$, where $$S\_i \subseteq X$$.

The following constraints apply:
1.  **Exclusivity:** No observation belongs to two clusters: $i \neq j \implies S\_i \cap S\_j = \emptyset$.
2.  **Exhaustivity:** Every observation belongs to a cluster: $\forall j \in [1,N] : \exists i \in [1,k] : \mathbf{x}\_j \in S\_i$.

The **centroid** ($$\mathbf{c}\_i$$) of a cluster is the average value of its observations, representing the core features of that group:

$$
\mathbf{c}\_i = \frac{1}{|S\_i|} \sum\_{\mathbf{x}\_i \in S\_i} \mathbf{x}\_i \in \mathbb{R}^p
$$

### Model Types

#### Centroid Models
These models determine a set of centroids $C = \{\boldsymbol{c}\_i\}\_{i=1}^k$ and assign each observation $\mathbf{x}\_j$ to the cluster $S\_i$ with the nearest centroid. 
* **Metric:** Proximity is determined by some distance function. Typical choice is Euclidean distance, but for molecular fingerprints e.g. **Tanimoto similarity** might be preferable.
* **Optimisation:** The algorithm minimises a within-cluster property, such as the sum of squared distances between points and their respective centroids.

#### Connectivity (Hierarchical) Models
Connectivity models use two main approaches to determine relationships:
* **Agglomerative (Bottom-up):** Starts with each point as its own cluster and iteratively merges the most similar pairs.
* **Divisive (Top-down):** Starts with one root cluster and partitions it based on high internal dissimilarity.

**Linkage Conditions** determine the distance between sets:
| Linkage | Description |
| :--- | :--- |
| **Ward** | Minimises the total within-cluster sum of squares. |
| **Single** | Distance between the two closest points of different clusters. |
| **Complete** | Distance between the two furthest points of different clusters. |

### Application to Atmospheric Molecules

In the context of **molecular fingerprints**, molecules in the same cluster tend to share similar structures. Clustering provides insights into structural properties such as:
* Symmetry and molecular weight.
* Presence or absence of functional groups.
* Aromaticity.
* Inferred properties (reactivity, flexibility, boiling point).

#### Centroids vs. Medoids
A **centroid** is a mathematical coordinate in hyperspace; it rarely maps to a single valid chemical graph. Instead, it reflects a **consensus pattern**. 
> **Example:** A centroid value of 0.8 at a specific bit position suggests that the corresponding structural fragment is conserved in 80% of the cluster members.

To find a concrete representative, we use a **medoid**—the actual molecule with the fingerprint closest to the centroid. 
* **High Similarity Clusters:** The medoid is a reliable proxy for the centroid.
* **High Variation Clusters:** The medoid may be distant from the mathematical average; exercise caution when using it to represent the entire group.

#### Interpreting Fingerprints
The interpretation of bit positions depends on the fingerprint type:
* **Structural Keys (e.g., MACCS, ATMOMACCS):** Straightforward. Bits correspond to the probability (bit vectors) or expected frequency (count vectors) of specific pre-defined substructures.
* **Hashed Fingerprints:** More complex. Since each bit is typically affected by multiple substructures, the centroid should be treated as a **structural motif** rather than a simple likelihood vector.

## <a id="clustering-metrics"></a> 2.1 Clustering Metrics

After partitioning, it is essential to evaluate how well the clusters fit the input data. Since the optimal number of clusters $k$ is rarely known a priori, we use validation techniques to compare different results.

### Types of Validation
* **Internal Validation:** Examines the partitioned data using only the data itself (e.g., via Cluster Validity Indices).
* **External Validation:** Compares results against a known "ground truth" partition.
* **Relative Validation:** Compares different results generated by the same algorithm using varying parameters.

### Mathematical Definitions
To evaluate these metrics, we define the following variables where $$d(\mathbf{x}\_i, \mathbf{x}\_j)$$ is the distance between two points:

* **Global Centroid:** $$\bar{X} = \frac{1}{N}\sum\_{i=1}^N \mathbf{x}\_i$$
* **Cluster Centroid:** $$\mathbf{c}\_i = \frac{1}{|S\_i|}\sum\_{\mathbf{x}\_j \in S\_i} \mathbf{x}\_j$$
* **Cluster Radius:** $$R\_i = \frac{1}{|S\_i|}\sum\_{\mathbf{x}\_j \in S\_i} d(\mathbf{x}\_j, \mathbf{c}\_i)$$
* **Cluster Variance:** $$\sigma^2\_i = \frac{1}{|S\_i|}\sum\_{\mathbf{x}\_j \in S\_i} d(\mathbf{x}\_j, \mathbf{c}\_i)^2$$
* **Intracluster Distance:** $$a(\mathbf{x}\_j \in S\_i) = \frac{1}{|S\_i| - 1}\sum\_{\mathbf{x}\_k \in S\_i, k\neq j} d(\mathbf{x}\_j, \mathbf{x}\_k)$$
* **Nearest-Cluster Distance:** $$b(\mathbf{x}\_j \in S\_i) = \min\_{l \neq i}\left\{\frac{1}{|S\_l|}\sum\_{\mathbf{x}\_k \in S\_l}d(\mathbf{x}\_j, \mathbf{x}\_k)\right\}$$

### Cluster Validity Indices (CVI)

Cluster validity is typically measured by balancing **cohesion** (intra-variance) and **separation** (inter-variance).

#### 1. Calinski-Harabasz (CH) Index
Estimates cohesion via weighted cluster variance and separation via the distance between cluster centroids and the global mean.
$$\text{CH}(S) = \frac{N-K}{K-1}\frac{\sum\_{S\_i \in S} |S\_i|d(\mathbf{c}\_i, \bar{X})^2}{\sum\_{S\_i\in S} |S\_i|\sigma^2\_i} \in [0, \infty]$$
* **Goal:** Higher values are better.
* **Indication:** Intracluster variance is significantly lower than intercluster separation.

#### 2. Davies-Bouldin (DB) Index
Estimates cohesion based on cluster radii and separation based on distances between centroids.
$$\text{DB}(S) = \frac{1}{K}\sum\_{S\_i \in S}\max\_{S\_j \in S \setminus S\_i} \left\{ \frac{R\_i + R\_j}{d(\mathbf{c}\_i, \mathbf{c}\_j)} \right\} \in [0, \infty]$$
* **Goal:** Lower values are better.
* **Indication:** Cluster radii are small relative to the distance between clusters.

#### 3. Silhouette Index
A summation-type index comparing the average distance to elements in the same cluster versus the nearest different cluster.
$$\text{Sil}(S) = \frac{1}{N}\sum\_{S\_i \in S}\sum\_{\mathbf{x}\_j \in S\_i} \frac{b(\mathbf{x}\_j) - a(\mathbf{x}\_j)}{\max \{a(\mathbf{x}\_j), b(\mathbf{x}\_j) \}} \in [-1,1]$$
* **Values near 1:** Points are perfectly clustered and well-separated.
* **Values near 0:** Points lie on decision boundaries.
* **Values near -1:** Observations are likely assigned to the wrong clusters.

### Application to Molecular Fingerprints

In the context of atmospheric molecules, these indices quantify how similar the molecular structures within a cluster truly are. 

| Metric | Interpretation for Molecules |
| :--- | :--- |
| **High CH** | The centroid is a statistically representative proxy for the molecules within the cluster. |
| **Low DB** | Molecular structures within each cluster are highly similar, and groups are distinct. |
| **High Sil** | Molecules are more structurally similar to their own cluster members than to any other group. |

> **Note on Distance Metrics:** While Euclidean distance is the standard convention, it may not be optimal for binary chemical fingerprints. More meaningful metrics, such as **Tanimoto similarity**, are often preferred for atmospheric chemistry applications.

## <a id="k-means"></a> 2.2 K-means

---

Given the set of observations, **K-Means** clustering aims to partition the observations into clusters such that the within-cluster sum of squares is minimized. In other words out objective is:

$$
\underset{\mathbf{S}}{\text{arg min}}\sum\_{i=1}^k\sum_{\mathbf{x} \in S\_i}\lVert\mathbf{x} - \boldsymbol{\mu}\_i\rVert^2,
$$

where $$\boldsymbol{\mu}\_i$$ is the centroid of the cluster $$S\_i$$.

## <a id="hac"></a> 2.3 HAC

---

\[Coming soon...\]

## <a id="birch"></a> 2.4 BIRCH

---

\[Coming soon...\]

## <a id="bisecting-k-means"></a> 2.5 Bisecting K-means

---

\[Coming soon...\]

---

# 3 Kernel Methods

\[Coming soon...\]

## 3.1 Linear kernel

\[Coming soon...\]

## 3.2 Polynomial kernel

\[Coming soon...\]

## 3.3 RBF kernel

\[Coming soon...\]

## 3.4 Cosine kernel

\[Coming soon...\]

## 3.5 Manhattan kernel

\[Coming soon...\]

## 3.6 Hamming kernel

\[Coming soon...\]

## 3.7 Tanimoto kernel

\[Coming soon...\]

---

# 4 Unsupervised Embedding

\[Coming soon...\]

## 4.1 Embedding Metrics

\[Coming soon...\]

## 4.2 PCA

\[Coming soon...\]

## 4.3 KPCA

\[Coming soon...\]

## 4.4 tSNE

\[Coming soon...\]

---

# 5 Semi-Supervised Embedding

\[Coming soon...\]

## 5.1 LSP

\[Coming soon...\]

## 5.2 cKPCA

\[Coming soon...\]

</main>

</div>