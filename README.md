# LearnPulse AI

LearnPulse AI is an experimental early-warning system that explores student engagement and academic outcome prediction using machine learning.

The project uses the Open University Learning Analytics Dataset (OULAD) to study student engagement, course outcomes, model evaluation, deployment, and responsible AI practices.

## Project Progress

| Week | Focus | Status |
|---|---|---|
| Week 1 | Project Planning and Documentation | Completed |
| Week 2 | Data Collection and Preprocessing | Completed |
| Week 3 | Model Development | See project files |
| Week 4 | Model Evaluation and Explainability | See project files |
| Week 5 | FastAPI, Docker Deployment and Monitoring | Completed |
| Week 6 | AI Ethics, Bias Analysis and Fairness Assessment | Completed |

## Repository Structure

- `week 1/` — Project planning and documentation.
- `week 2/` — Dataset notes, preprocessing notebook, and processed dataset.
- `week 5/` — FastAPI application, Docker configuration, model artifact, and deployment report.
- `week 6/` — Ethics and fairness report, audit scripts, evaluation metrics, and visualizations.

## Technologies

- Python, pandas, NumPy
- scikit-learn and Random Forest
- FastAPI and Uvicorn
- Docker
- Matplotlib
- Git and GitHub

## Dataset

The project uses the Open University Learning Analytics Dataset (OULAD). Raw dataset files are not included in the repository. Follow the instructions in `week 2/DATASET_SOURCE.md` to obtain the dataset and configure the local project.

## Evaluation and Responsible Use

The initial Week 6 fairness audit reported 69.17% test accuracy. The day-60 temporal audit reported 57.14% accuracy and 11.2% recall for the Withdrawn class.

These results demonstrate limitations in identifying some student outcomes. The model is experimental and requires further validation, fairness analysis, and monitoring. Predictions must support constructive human review, not automated or punitive decisions about students.

## Running the Week 5 API

From the project root, build and run the Docker image:

```bash
docker build -t learnpulse-api:1.0 "./week 5"
docker run --rm -p 8000:8000 learnpulse-api:1.0
```

Then open `http://127.0.0.1:8000/docs` to explore the API.

## Important Notes

The Week 6 audit scripts may require path configuration on another computer. The project also needs additional validation before any real-world student-support use.