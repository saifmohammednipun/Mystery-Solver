# ClueChain: A Bayesian Network-Based Mystery Solver Application

## Overview

**ClueChain** is a mystery solver application that uses Bayesian networks to solve fictional mystery scenarios. The application allows users to input various clues and evidence related to a crime and then applies Bayesian inference to calculate the probabilities of different suspects being guilty. The system provides an interactive and data-driven way to deduce the most likely suspect based on the provided evidence.

## Features

- **Clue Input**: Users can provide various clues or pieces of evidence that may relate to the crime.
- **Bayesian Inference**: The system applies Bayesian networks to calculate the likelihood of suspects being guilty based on inputted clues.
- **Interactive UI**: The application provides an easy-to-use interface for adding clues, viewing results, and understanding the logic behind the probability calculations.
- **Mystery Scenarios**: The app can be used for various fictional mystery scenarios, making it adaptable to a variety of situations.

## Usage

To run the project, execute the `main.py` file:

```bash
python main.py
```

You will be prompted to enter clues and evidence for the mystery, and the application will use Bayesian inference to calculate and display the probability of different suspects.

## Project Structure

The repository is organized as follows:

```
ClueChain/
│
├── main.py                  # Main code file to run the project
├── README.md                # Project documentation (this file)
├── requirements.txt         # List of libraries and dependencies
│
├── data/                    # Subfolder containing datasets
│   └── sample_data.json     # Example dataset used for testing
│
├── support/                 # Subfolder containing other code files
│   ├── bayesian_network.py  # Code for Bayesian network logic
│   ├── clue_input.py        # Code to handle clue input from the user
│   └── probability_calculations.py # Code for calculating probabilities
│
└── others/                  # Subfolder for project-related documents
    ├── final_presentation.pptx  # Final presentation file
    ├── final_report.pdf        # Final project report
    ├── update_presentation.pptx  # Updated presentation file
    ├── update_report.pdf        # Updated project report
    └── demo_video.mp4          # One-minute demo video of the project
```

## 🧠 Technologies Used

- [Streamlit](https://streamlit.io/) - Web application interface
- [pgmpy](https://pgmpy.org/) - Probabilistic graphical models
- [Graphviz](https://graphviz.org/) - Visualization of network structures
- Python 3.8+


## Requirements

This project relies on the following Python libraries:

- `numpy` - For numerical calculations
- `pgmpy` - For creating and manipulating Bayesian networks
- `networkx` - For creating and manipulating Bayesian networks
- `matplotlib` - For visualizing the Bayesian network and results
- `json` - For handling input and output in JSON format

To install all the necessary dependencies, simply run:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions to enhance the ClueChain project. If you would like to contribute, follow these steps:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Open a pull request to merge your changes into the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- Thanks to all contributors and open-source libraries that helped in the development of ClueChain.
- Special thanks to [Dr. Mohammad Shifat-E-Rabbi (MSRb)](https://sites.google.com/view/m-shifat-e-rabbi).