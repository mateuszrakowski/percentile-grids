# Percentile Grids

A web-based application for calculating and visualizing percentile grids for brain structure volumes against a normative reference dataset. This tool is designed to help researchers and clinicians analyze brain development and identify potential abnormalities.

## ✨ Features

-   **Upload Your Data**: Supports both CSV and Excel file formats for input.
-   **Automated Age Calculation**: Automatically calculates subject age from date of birth and scan date.
-   **GAMLSS Modeling**: Utilizes Generalized Additive Models for Location, Scale, and Shape (GAMLSS) to generate accurate percentile values.
-   **Utilizes the R packages**: GAMLSS R packages are used to prepare correct results (the application uses the `rpy2` library to interface with R).
-   **Interactive Web Interface**: Built with Streamlit for a user-friendly experience.
-   **Reference Dataset Viewer**: A dedicated page to explore the underlying reference dataset.

## 🚀 Getting Started

This application is designed to be run using Docker to ensure a consistent and reliable environment.

### Prerequisites

-   Docker installed on your system.

### Running with Docker

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd percentile-grids
    ```

2.  **Build the Docker image:**
    From the root of the project directory, run the following command. This will build the image and tag it as `percentile-grids-app`.
    ```bash
    docker build -t percentile-grids-app .
    ```

3.  **Run the Docker container:**
    This command starts the container, maps port `8080` on your local machine to port `8080` inside the container, and mounts a local `data` directory for any data you might want the application to access.

    Make sure to run the docker command inside the cloned repository, as it will create the `data/` directory to store the models.
    ```bash
    docker run -p 8080:8080 percentile-grids-app
    ```

4.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8080`.

## 📁 Input Data Format

Currently the input data to the application is rigidly fixed and is generated from other software.
An example of the input data structure can be found in `scripts/original_mock.csv`.

##  Project Structure

-   `grids/`: The main source code for the application.
    -   `data_processing/`: Handles input file reading and data preparation.
    -   `engine/`: Contains the GAMLSS modeling and prediction logic.
    -   `web_interface/`: Defines the Streamlit user interface pages.
    -   `resources/`: Stores static resources like brain structure lists.
    -   `main.py`: The entry point for the Streamlit application.
-   `tests/`: Contains all the tests for the project.
-   `scripts/`: Includes helper scripts, such as for creating mock data.
-   `Dockerfile`: Defines the container for the application.
-   `requirements.txt`: Lists the Python dependencies for the application.
-   `requirements-dev.txt`: Lists additional dependencies for development and testing.