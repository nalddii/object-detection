## Project Instructions

Follow these steps to set up and run the project:

### 1. Install Requirements

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Prepare Input Images

- Place an image of an apple in the `input_classify` directory.
- Place an image of an oil-palm tree in the `input_count` directory.

### 3. Run `count.py`

Run the following command to process the image in the `input_count` directory:

```bash
python count.py
```

- **Output**: The processed results will be saved in the `output_count` directory.
- **Note**: This step may take some time depending on the image size.

### 4. Run `classify.py`

Run the following command to classify the image in the `input_classify` directory:

```bash
python classify.py
```

- **Output**: The processed results will be saved in the `output_classify` directory.

### 5. Finish

Once both scripts have completed, the outputs will be available in their respective directories.
