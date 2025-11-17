# Assignment 1 - Decoding Controls and Structured Output

This repository contains code for working with Hugging Face Transformers models, specifically focusing on decoding controls and structured output generation.

## Problem 1: Decoding Controls and Structured Output

Demonstrates how decoding parameters (temperature, top-k, top-p) affect diversity, repetition, and structure in text generation using the `distilgpt2` model.

## Setup

Install required dependencies:

```bash
pip install transformers torch
```

## Files

- `problem1.py` - Main script for loading model and generating samples
- `check_gpu.py` - Utility script to check GPU availability

## Usage

Run the main script:

```bash
python problem1.py
```

## Model

- **Model**: distilgpt2 from Hugging Face
- **Task**: JSON extraction from purchase requests

