<!-- LOGO & TITLE -->
<div align="center">
  <a href="https://github.com/mabaan/Anime-Recommendation-Model">
    <img src="https://ani-github.github.io/animegifs/full-metal-alchemist/AWESOME.gif" alt="Logo" width="400">
  </a>
  <h3 align="center">Anime Recommendation Model</h3>

  <p align="center">
    A hybrid ML-powered recommendation system combining collaborative filtering and content-based techniques to suggest anime to keep users entertained.
    <br />
    <br>
    <a href="https://github.com/mabaan/Anime-Recommendation-Model/main"><strong>Explore the code </strong></a><br><br>
    <a href="https://github.com/mabaan/Anime-Recommendation-Model/tree/main/Documentation"><strong>View Detailed Documentation »</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#to-run">To Run</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project
This repository presents a **multi-stage hybrid anime recommendation engine** that dynamically adapts to each user’s journey. By weaving together three core approaches—non‑personalized popularity, content‑based filtering, and collaborative filtering—our system ensures both **relevant** and **diverse** anime suggestions at every stage:
<img align="right" src="https://github.com/user-attachments/assets/6da05881-ddfc-4661-a2cb-661d91287993" alt="Photo" width=400px ><br /><br />

- **Stage 1 (Cold Start - Non Personalized)**: New users see top‑ranked anime via a *hybrid popularity model* combining Bayesian smoothing (to counter small‑sample bias) and weighted scoring (to balance quality vs. popularity). This surface universally acclaimed titles and rising gems immediately.
- **Stage 2 (Content‑Based Personalization)**: After initial interactions (genre selections or clicks), the engine switches to a *hybrid content model*—merging **TF‑IDF cosine similarity** on plot synopses with **Jaccard similarity** on genre tags. This prioritizes narrative richness while preserving thematic consistency.
- **Stage 3 (Collaborative Filtering)**: Once sufficient ratings exist, recommendations pivot to a *hybrid CF model*, blending **item‑based CF** (70% weight for stability and lower RMSE) with **user‑based CF** (30% weight for personal nuance). This final phase optimizes predictive accuracy and user satisfaction.

Our evaluation shows **item-based CF achieves RMSE = 1.21/10**, and the overall pipeline maintains recommendation diversity without sacrificing relevance. Detailed EDA, methodology, and performance charts are available in the project report and presentation.

## Built With
- **Python** &mdash; Pandas, NumPy, scikit-learn, tenacity
- **Flask** &mdash; Web API and static file server
- **React** &mdash; Frontend library (optional)
- **HTML/CSS/JavaScript** &mdash; UI components


## Getting Started
Follow these steps to get a local copy up and running.

### Prerequisites
- Python 3.7+
- Node.js & npm

### Installation
```sh
# Clone the repository
git clone https://github.com/mabaan/Anime-Recommendation-Model.git
cd Anime-Recommendation-Model
# Install Python dependencies
pip install -r requirements.txt
```

## To Run
<div align="center">
  <!-- Sample anime poster previews -->
  <img src="https://github.com/user-attachments/assets/ececf4aa-27f0-4e1a-9085-7cd8d1e0a4f3" width="500"/>
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/a1e31514-b35b-4659-9806-ed11bda2c897" width="500"/>
  &nbsp;&nbsp;
   <img src="https://github.com/user-attachments/assets/8c4a9901-cc2f-41b2-9ebb-a5334a1f6b55" width="500"/>
  &nbsp;&nbsp;
</div>

**Backend** (Flask):

```sh
# Ensure anime_df_final.pk and rating_df_final.pk are in the root
python app.py
```

The Flask app will load two pickled DataFrames:
```python
anime_df_clean = pd.read_pickle("anime_df_final.pk")
rating_df_clean = pd.read_pickle("rating_df_final.pk")
```

**Frontend** (React static build):
```sh
cd frontend
npm install
npm run build
``` 
Files are generated in `frontend/build` and served by Flask.


## Contributing
1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request


## License
Distributed under the MIT License. See `LICENSE` for details.


## Contact
Mohammed Abaan: <br>
<a href="mailto:abaan7500@gmail.com">
  <img src="https://img.shields.io/badge/Gmail-d5d5d5?style=for-the-badge&logo=gmail&logoColor=0A0209" alt="Email Abaan" />
</a>
<br><br>
Ahmed Mehaisi: <br>
<a href="mailto:b00094989@aus.edu">
  <img src="https://img.shields.io/badge/Gmail-d5d5d5?style=for-the-badge&logo=gmail&logoColor=0A0209" alt="Email Ahmed" />
</a>
<br><br>
Project Link: [https://github.com/mabaan/Anime-Recommendation-Model](https://github.com/mabaan/Anime-Recommendation-Model)

## Acknowledgments
- [Pandas Documentation](https://pandas.pydata.org/)
- [Flask Official Docs](https://flask.palletsprojects.com/)
- [Anime Metadata Source](https://www.kaggle.com/datasets)

<!-- SHIELDS & LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/mabaan/Anime-Recommendation-Model.svg?style=for-the-badge
[contributors-url]: https://github.com/mabaan/Anime-Recommendation-Model/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mabaan/Anime-Recommendation-Model.svg?style=for-the-badge
[forks-url]: https://github.com/mabaan/Anime-Recommendation-Model/network/members
[stars-shield]: https://img.shields.io/github/stars/mabaan/Anime-Recommendation-Model.svg?style=for-the-badge
[stars-url]: https://github.com/mabaan/Anime-Recommendation-Model/stargazers
[issues-shield]: https://img.shields.io/github/issues/mabaan/Anime-Recommendation-Model.svg?style=for-the-badge
[issues-url]: https://github.com/mabaan/Anime-Recommendation-Model/issue
