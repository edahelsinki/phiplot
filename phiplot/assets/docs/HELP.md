# Quick Start Guide

---

## 1. Views and Layout

<div style="font-size: 1.75em">

The application provides three distinct views to analyse data at increasing levels of detail. You can switch between them using the **View Selector** in the top menu:

- **Data Summary** – Compute and plot summary statistics for the entire dataset.
- **Clustering** – Group molecular subsets using various algorithms and visualise them via dimensionality reduction.
- **Embedding** – Perform high-fidelity projections with interactive constraints and detailed data point inspection.

</div>

### Interface Structure

<div style="font-size: 1.75em">

Each view is organised into four functional areas:
- **Top Menu Row** – Context-specific actions and controls for the active view.
- **Left Column** – Detailed information related to the active view. 
- **Center Column** – The primary interactive display area.
- **Right Column** – Global status information, including the active collection and applied filters (shared across all views).

</div>

---

## 2. Accessing Data

### 2.1 Connecting to the Server

<div style="font-size: 1.75em">

Navigate to **Data** → **Connect to Server**. A **green** notification and indicator signify a successful connection; **red** indicates a connection failure.

</div>

### 2.2 Selecting a Collection

<div style="font-size: 1.75em">

Once connected, go to **Data** → **Select Collection**. 
1. Choose a database from the server.
2. Select a specific collection within that database.
3. Click **Connect to Collection** to confirm.

</div>

### 2.3 Collection Settings

<div style="font-size: 1.75em">

Before fetching data, verify settings via **Data** → **Collection Settings**. Ensure the following fields are mapped correctly:
- **Index Field:** The unique identifier for molecules.
- **SMILES Field:** The string used for generating fingerprints and 2D structures.
- **Fetch Fields:** Specific document fields to be retrieved.

> ⚠️ **NOTE:** *The tool makes initial guesses for the collection settings. Always double-check them after connecting to a new collection.*

</div>

### 2.4 Fetching Data

<div style="font-size: 1.75em">

Navigate to **Data** → **Fetch Data**. Choose from five retrieval methods:
- **Random Sample** – Specify a set number of molecules.
- **By Filters** – Fetch molecules meeting specific criteria (see Section 4).
- **Index Range** – Define lower and upper numerical bounds.
- **Index Set from File** – Upload a `.txt` file with one index per line.
- **All** – Retrieve the entire collection.

> ⚠️ **NOTE:** *For optimal performance it is recommended to keep fetched samples under 10,000 data points.*

</div>

---

## 3. Fingerprinting


<div style="font-size: 1.75em">

To perform chemical analysis, you must convert SMILES strings into numerical vectors. 
1. Navigate to **Data** → **Generate Fingerprints**.
2. **Optional:** Click **Fingerprint Parameters** to adjust algorithm settings.
3. Click **Generate Fingerprints**. 

Two progress bars will appear: one for 2D structure generation and one for fingerprint calculation. A **green** notification confirms completion.

</div>

---

## 4. Filtering

### 4.1 Applying a Filter

<div style="font-size: 1.75em">

Go to **Filter** → **Apply Filter**. Select a feature and define the condition:
- **Range** – Inclusive numerical bounds.
- **Unary** – Exclusive comparisons (e.g., $X > 5$).
- **Categorical** – Exact matches for text-based data.

</div>

### 4.2 Managing Filters

<div style="font-size: 1.75em">

- **Visual Cue:** Filtered-out points are **faded** in the embedding view. They still occupy space in the calculation unless removed.
- **Remove Individual:** Click the **×** on the filter card in the **Right Column**.
- **Clear All:** Navigate to **Filter** → **Clear All Filters**.
- **Purge Data:** To stop filtered points from affecting embedding calculations, select **Filter** → **Remove Filtered**.

</div>

---

## 5. Data Summary

<div style="font-size: 1.75em">

The **Data Summary** view is the starting point for exploratory analysis.

</div>

### 5.1 Summarising Data

<div style="font-size: 1.75em">

Navigate to **Summarise** and choose the appropriate field type:

- **Numerical Fields:** Options include bin count selection for histograms, filter toggle, relative frequency toggle, and Kernel Density Estimate (KDE) overlay toggle.
- **Categorical Fields:** Options include comparison field selection, filter toggle, relative frequency toggle, and notched boxes toggle.

</div>

### 5.2 Appearance & Search

<div style="font-size: 1.75em">

- **Search:** Use the **Search Bar** (top right of the Centre Column) to find a molecule by index. 
- - **Appearance:** Use the **Appearance** → **Colour Settings** to modify plot colours or **Appearance** → **Toggle Legends** to toggle the legends on and off.

> 💡 **KDE:** *The underlying continuous probability density function is approximated from discrete binned data by treating bin centres as observations weighted by their counts.*

> 💡 **Notched Boxes:** *In comparison plots, notch boundaries are defined as $$\pm \frac{1.58 \cdot \text{IQR}}{\sqrt{n}}$$, where $$\text{IQR}$$ is the interquartile range and $$n$$ is the sample size.*

</div>

---

## 6. Clustering

<div style="font-size: 1.75em">

The **Clustering** view allows you to group molecules based on chemical similarity.

</div>

### 6.1 Running a Cluster Analysis

<div style="font-size: 1.75em">

> ⚠️ **NOTE:** *Before running the analysis, make sure data has been fetched (see Section 2.4) and fingerprints generated (see Section 3).*

Navigate to **Cluster** and select an algorithm (e.g., KMeans, BIRCH). 
1. Select the **Fingerprint** type to be analysed.
2. Select an **Embedding Algorithm** for the 2D visualisation (e.g., PCA, t-SNE).
3. Adjust **Hyperparameters** if desired by clicking either **Clustering Hyperparams** or **Embedding Hyperparams**.
4. Click **Cluster**.

> ⚠️ **NOTE:** *For optimal clustering performance it is recommended to keep the sample under 10,000 data points at this stage.*

</div>

### 6.2 New feature from Clusters

<div style="font-size: 1.75em">

To use your clusters as filters for the embedding view, go to **Clustering** → **Generate Features from Labels**. This creates a new categorical feature accessible in the **Filter** menu.

</div>

### 6.3 Appearance & Search

<div style="font-size: 1.75em">

- **Search:** Enter an index in the **Search Bar** (top right of the Centre Column); the molecule will be highlighted with a **fading magenta disk**.
- **Appearance:** Use the **Appearance** → **Colour Settings** to modify plot colours or **Appearance** → **Toggle Legend** to toggle the legend on and off.

> 💡 **Clustering Algorithms:** *You can choose from several partitioning methods, including Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH), K-means, and Bisecting K-means. These algorithms analyse high-dimensional fingerprints to organise molecules into discrete groups, or 'clusters', where points within a group share a higher degree of similarity with one another than with those in external groups. Because each algorithm employs different mathematical heuristics, such as hierarchical branching or centroid-based partitioning, they provide distinct yet complementary perspectives on the underlying structure of your molecular data.*

> 💡 **Embedding algorithms:** *The application provides several unsupervised dimensionality reduction algorithms, including Principal Component Analysis (PCA), Independent Component Analysis (ICA), Kernel Principal Component Analysis (KPCA), Locally Linear Embedding (LLE), Multidimensional Scaling (MDS), and t-Distributed Stochastic Neighbour Embedding (t-SNE). These methods project the high-dimensional molecular fingerprints that have been clustered in the original data space into the 2D plane. Because each algorithm prioritises different mathematical properties, such as global variance or local neighbourhood preservation, using them in tandem offers a more comprehensive understanding of the clustered chemical space.*

</div>

---

## 7. Embedding

<div style="font-size: 1.75em">

The **Embedding** view allows you to interactively constrain and explore molecules based on chemical similarity.

</div>

### 7.1 Algorithm Types

<div style="font-size: 1.75em">

> ⚠️ **NOTE:** *Before starting the embedding process, make sure data has been been fetched (see Section 2.4) and fingerprints generated (see Section 3).* 

Select a method via the **Algorithm** menu:
- **Static:** Projections like PCA, LLE, Isomap, ICA, t-SNE, or MDS.
- **Interactive:** Projections that respond to user constraints (**cKPCA**, **LSP**).

</div>

### 7.2 Interactive Constraints

<div style="font-size: 1.75em">

You can "guide" the layout of interactive embeddings:
- **Control Points:** Click a point and start dragging it while keeping the left mouse button pressed. Alternatively use **Constraints** → **Add Control Point**.
- **Link Constraints:** Select two points and press **`m`** (Must-link) to pull them together, or **`c`** (Cannot-link) to push them apart. Alternatively use **Constraints** → **Add Must-Link** or **Add Cannot-Link**.
- **Removing Constraints:** Click the **×** on the constraint’s card in the left column or use the **Constraints** menu.

⚠️ **NOTE:** *For optimal interactive embedding performance it is recommended to keep the sample under 1,000 data points at this stage.*

</div>

### 7.3 Appearance & Search

<div style="font-size: 1.75em">

- **Search:** Enter an index in the **Search Bar** (top right of the Centre Column); the molecule will be highlighted with a **fading magenta disk**.
- **Colouring by Features:** Use **Appearance** → **Colour Datapoints by Feature** to apply heatmaps based on numerical data.
- **Appearance:** Use the **Appearance** → **Colour Settings** to modify plot colours or **Appearance** → **Toggle Legend** to toggle the legend on and off.

> 💡 **Embedding Algorithms:** *The application provides several unsupervised dimensionality reduction algorithms, including Principal Component Analysis (PCA), Independent Component Analysis (ICA), Kernel Principal Component Analysis (KPCA), Locally Linear Embedding (LLE), Multidimensional Scaling (MDS), and t-Distributed Stochastic Neighbour Embedding (t-SNE). These methods project complex, high-dimensional molecular fingerprints into the 2D plane. Static embeddings provide a fixed global or local representation of the data structure, while interactive embeddings (cKPCA, LSP) allow for real-time adjustment based on user-defined constraints. Because each algorithm prioritises different mathematical properties, such as global variance or local neighbourhood preservation, using them in tandem offers a more comprehensive understanding of the chemical space.*

</div>

---