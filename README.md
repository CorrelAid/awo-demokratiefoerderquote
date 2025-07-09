## Setup

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Dev Setup

1. Clone this repository
2. `uv sync`
3. `uv run pre-commit install`

## AIM UI

Unfortunately, it can cause problems to run the aim ui and serve at the same time when they have mounted to the same repo. Therefore:
```
kubectl scale --replicas=0 deployment/aim-ui-deployment
```

## License
### Code 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

see `LICENSE-CODE`

### Data 

We refer to the [imprint of foerderdatenbank.de](https://www.foerderdatenbank.de/FDB/DE/Meta/Impressum/impressum.html) of the Federal Ministry for Economic Affairs and Climate Action which indicates [CC BY-ND 3.0 DE](https://creativecommons.org/licenses/by-nd/3.0/de/deed.de) as the license for all texts of the website. The dataset provided in this repository transfers information on each funding program into a machine-readable format.


