<img src="phiplot/assets/media/banner.png" alt="PhiPlot banner" width="250"/>

# Welcome!

PhiPlot is a web-based, interactive Exploratory Data Analysis (EDA) environment designed to explore data about atmospherically relevant molecules. PhiPlot leverages knowledge-based dimensionality reduction to support hypothesis generation, informed subsetting of molecules, and uncovering meaningful patterns within these complex molecular datasets.

The application provides an easy-to-use, accessible interface. The user can get an overview of the data by accessing summary statistics of the available covariates, possibly with filters applied. The user can then cluster the molecules. The cluster labels can be used as new features, e.g., to further filter the data. Finally, a subset of the data can be embedded in a two-dimensional plane, with the ability to interactively apply embedding constraints.

To get you started, you can access the *Quick Start Guide* by clicking the **Help** button at the top right corner of the title row.

> 💡 *The source code of the application can be found on [GitHub](https://www.github.com/edahelsinki/phiplot)*

## Acknowledgments
The interactive embedding feature of PhiPlot is motivated by [InVis](https://link.springer.com/chapter/10.1007/978-3-642-40994-3_52) application by Paurat and Gärtner, and [InVis 2.0](https://link.springer.com/chapter/10.1007/978-3-031-70371-3_34) application by Chen and Gärtner. Our specific implementation is inspired by the InVis 2.0 tool. The algorithmic principle behind the interactive embedding is based on [Knowledge-Based Kernel PCA](https://link.springer.com/chapter/10.1007/978-3-662-44851-9_32) introduced by Oglic, Paurat and Gärtner. The [ATMOMACCS](https://arxiv.org/abs/2510.20465) fingerprinting algorithm is provided by Lind and Rinke. The application is running on Rahti container orchestration service, provided by [CSC--IT Center for Science](https://csc.fi/en/).