<img src="./banner.png" alt="drawing" width="350"/>

# PhiPlot

**PhiPlot** is a web-based, interactive Exploratory Data Analysis (EDA) environment designed to explore data about atmospherically relevant molecules [1, 2, 3]. PhiPlot leverages knowledge-based dimensionality reduction to support hypothesis generation, informed subsetting of molecules, and uncovering meaningful patterns within these complex, molecular datasets.

The application provides an easy-to-use, accessible interface. The user can get an overview of the data by accessing summary statistics of the available covariates, possibly with filters applied. The user can then cluster the molecules. The cluster labels can be used as new features, e.g., to further filter the data. Finally, a subset of the data can be embedded in a two-dimensional plane, with the ability to interactively apply embedding constraints.

The deployed and running application can be found at: <https://phiplot-vilma-mongodb.2.rahtiapp.fi/>. The **Quick Start Guide** for using the application can be accessed by pressing the `Help?` button at the top right corner of the interface. 

The source code has been published under the [MIT license](./LICENSE).

### Acknowledgments
The interactive embedding feature of PhiPlot is motivated by the [InVis](https://link.springer.com/chapter/10.1007/978-3-642-40994-3_52) application by Paurat and Gärtner [4], and the [InVis 2.0](https://link.springer.com/chapter/10.1007/978-3-031-70371-3_34) application by Chen and Gärtner [5]. Our specific implementation is inspired by the InVis 2.0 application. The algorithmic principle behind the interactive embedding is based on [Knowledge-Based Kernel PCA](https://link.springer.com/chapter/10.1007/978-3-662-44851-9_32) introduced by Oglic, Paurat and Gärtner [6]. The [ATMOMACCS](https://arxiv.org/abs/2510.20465) fingerprinting algorithm is provided by Lind and Rinke [7]. The application is running on Rahti container orchestration service, provided by [CSC--IT Center for Science](https://csc.fi/en/).

## Local testing with Docker

We recommend using the deployed version, which requires no installation or configuration from the user, at: <https://phiplot-vilma-mongodb.2.rahtiapp.fi/>. However, if the site is unavailable, or you require a local testing environment, please use the following deployment steps.

Ensure you have [Docker Engine](https://docs.docker.com/engine/install/) and the [Compose plugin](https://docs.docker.com/compose/install/linux/) installed. Verify by running:

```bash
docker compose version
```

### Running with a local database
Ensure ports `5006` (used by the web app) and `27017` (used by the database) are free on your machine before running. To build the application accompanied by a local database instance (using sample data from the `./gecko` directory), run:

```bash
docker compose --profile local-db up --build
```

Access the app at <http://localhost:5006/>.

> 💡 This command automatically initialises a local MongoDB instance, imports the molecule datasets, and launches the application. The local database instance created contains a subset of the full data. If you want the local application to access the full data, follow the instructions below.

### Running with remote database access

The full dataset is available via a remote database instance. To use this:
1. Get Credentials: Contact us via [email](mailto:matias.loukojarvi@helsinki.fi) for read-only connection strings.
2. Setup Env: Create a `.env` file in the project root and add the connection string as variable `MONGO_URI=mongodb://username:password@hostname`
3. Free up ports: Ensure port `5006` is free on your machine before running
4. Launch: Run the standard compose command:

```bash
docker compose up --build
```

Access the app at <http://localhost:5006/>.

> 💡 In this mode, the local database container will not start and the application will target the remote URI instead. 

### Stopping the application

To shut down all services and clean up containers, run:

```bash
docker compose --profile local-db down
```

## References

[1] Besel, V. (2023). *GeckoQ: Atomic structures, conformers and thermodynamic properties of 32k atmospheric molecules*. University of Helsinki, Institute for Atmospheric and Earth System Research. https://doi.org/10.23729/44e30aaa-ec3c-49ae-90e2-bd598a7262fe  

[2] Franzon, L., Camredon, M., Valorso, R., Aumont, B., & Kurtén, T. (2024). *Ether and ester formation from peroxy radical recombination: A qualitative reaction channel analysis*. *Atmospheric Chemistry and Physics, 24*(20), 11679–11699. https://doi.org/10.5194/acp-24-11679-2024

[3] Kähärä, J., Franzon, L., Ingram, S., Myllys, N., Kurtén, T., & Vehkamäki, H. (2025). Enhanced configurational sampling methods reveal the importance of molecular stiffness for clustering of oxygenated organic molecules. Physical Chemistry Chemical Physics, 27(43), 23410–23420. https://doi.org/10.1039/D5CP01931A

[4] Paurat, D., & Gärtner, T. (2013). *InVis: A tool for interactive visual data analysis*. In H. Blockeel, K. Kersting, S. Nijssen, & F. Železný (Eds.), *Machine learning and knowledge discovery in databases* (Lecture Notes in Computer Science, Vol. 8190, pp. 672–676). Springer. https://doi.org/10.1007/978-3-642-40994-3_52  

[5] Chen, F., & Gärtner, T. (2024). *Scalable interactive data visualization*. In A. Bifet et al. (Eds.), *Machine learning and knowledge discovery in databases: Research track and demo track* (pp. 429–433). Springer Nature. https://doi.org/10.1007/978-3-031-70371-3_34  

[6] Oglic, D., Paurat, D., & Gärtner, T. (2014). *Interactive knowledge-based kernel PCA*. In T. Calders, F. Esposito, E. Hüllermeier, & R. Meo (Eds.), *Machine learning and knowledge discovery in databases* (pp. 501–516). Springer. https://doi.org/10.1007/978-3-662-44851-9_32  

[7] Lind, L., Sandström, H., & Rinke, P. (2025). *An interpretable molecular descriptor for machine learning predictions in atmospheric science*. *arXiv preprint* arXiv:2510.20465. https://arxiv.org/abs/2510.20465