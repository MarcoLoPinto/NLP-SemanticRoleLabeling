# NLP - Semantic Role Labeling

Semantic Role Labeling is the task of of automatically extract predicate-argument structure from a given sentence.

The selected predicate defines an action/event with its arguments, each of them with a specific role. 
An example is reported below:
<p align="center">
    <img width="360" src="./imgs/SRL_example.png" alt="Material Bread logo">
</p>

The UniteD-SRL dataset  (Tripodi et al., 2021) will be used for this task. 

The **report.pdf** file describes the solution adopted, compared with alternative approaches to said problem.

This is the second homework of the NLP 2022 course at Sapienza University of Rome.

#### Instructor

* **Roberto Navigli**
  * Webpage: [http://www.diag.uniroma1.it/~navigli/](http://www.diag.uniroma1.it/~navigli/)

#### Teaching Assistants

* **Andrei Stefan Bejgu**
* **Riccardo Orlando**
* **Alessandro Scirè**
* **Simone Tedeschi**

#### Course Info

* [http://naviglinlp.blogspot.com/](http://naviglinlp.blogspot.com/)

## Requirements

* Ubuntu distribution
  * Either 22.4 or the current LTS (20.04) are perfectly fine
  * If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project

## Setup Environment

As mentioned in the slides, differently from previous years, this year we will be using Docker to remove any issue pertaining your code runnability. If test.sh runs
on your machine (and you do not edit any uneditable file), it will run on ours as well; we cannot stress enough this point.

Please note that, if it turns out it does not run on our side, and yet you claim it run on yours, the **only explanation** would be that you edited restricted files,
messing up with the environment reproducibility: regardless of whether or not your code actually runs on your machine, if it does not run on ours,
you will be failed automatically. **Only edit the allowed files**.

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding. For those who might be
unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependencies to run. We will be using conda to create the environment for this client.

```bash
conda create -n nlp2022-hw2 python=3.9 -y
conda activate nlp2022-hw2
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp2022-hw2
bash test.sh data/EN/dev.json EN
```

`test.sh` takes two mandatory arguments (in this particular order):

* the path to the file you want to evaluate: you can replace `data/EN/dev.json` to point to a different file, as far as the target file has the same format
* the language of the file you want to evaluate: the options are `EN` (English, mandatory), `FR` (French, optional) and `ES` (Spanish, optional)

If you hadn't changed `hw2/stud/implementation.py` yet when you run test.sh, the scores you just saw describe how a random baseline
behaves. To have `test.sh` evaluate your model, follow the instructions in the slides.
