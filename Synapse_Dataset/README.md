# Synapse | PARSEC 6.0

This dataset was created for **Synapse: The Neurotech Challenge** at **PARSEC 6.0, IIT Dharwad**. It contains multi-session, multi-subject sEMG recordings of five hand gestures.

## Overview

The dataset includes **8-channel surface EMG signals** recorded from **25 subjects**, each performing **five gestures** across **three separate recording sessions**. Every gesture has multiple trials per subject.

## Directory Structure

The dataset is organized into three top-level folders:

- `Session1`
- `Session2`
- `Session3`

Each session directory contains **25 subject folders**.

### Subject Folders

Subject folders follow the naming pattern: `session{id}_subject{id}`. Each subject folder contains multiple `.csv` files.

### CSV File Format

Files are named as: `gesture{id}_trial{id}.csv` 

Each CSV contains:

- **5 seconds** of sEMG data  
- **8 channels**
- Sample rate: **512 Hz**
- Time integrity preserved (no dropped samples)

## Notes the Dataset

- All 8 sEMG electrodes where placed on forearm of subject.
- Each `.csv` file contains 5 seconds of data when the subject perform the gesture.
- Each gesture was repeated for 7 trials.
- Partipants will have to infer dependant label from file name for training their models.
- Each Session corrosponds to a different day the data recorded

> Effectively, participants are given 7 trials of 5 gestures, performed by 25 participants each across 3 days.

## Disclaimer
Organizers reserve the right to modify, restrict, or clarify any aspect of the dataset or challenge rules at any point during the event.

For any queries: Contact us via

- Email: `outreach.parsec@iitdh.ac.in`



