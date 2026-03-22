# Reproducing the CLRerNet Paper on Your Laptop

### Part 1: The "From Scratch" Master Setup Guide
*If you are starting fresh on a new machine, these are the exact steps you took to get here.*

**1. Prerequisites:**
* Install **Docker Desktop** (or Docker Engine on Linux).
* Ensure **NVIDIA Drivers** are installed on the host.
* (If on Windows) Ensure Docker Desktop has WSL2 integration enabled.

**2. Clone the Repository:**
```bash
git clone https://github.com/hirotomusiker/CLRerNet.git
cd CLRerNet
```

**3. Optimize the Configuration (`docker-compose.yaml`):**
To save time and RAM, edit the YAML file before building:
* Change `TORCH_CUDA_ARCH_LIST` to match the target GPU (e.g., `"8.6"` for RTX 3050).
* Change `shm_size` from `"16gb"` to `"8gb"` (or whatever leaves enough RAM for the host OS).

**4. Build the Docker Image:**
*This downloads PyTorch, CUDA, and compiles the core libraries.*
```bash
docker-compose build --build-arg UID=$(id -u) dev
```

**5. Start the Container & Fix NMS:**
*The Non-Maximum Suppression (NMS) layer must be compiled manually inside the running container.*
```bash
docker-compose run --rm dev
cd libs/models/layers/nms/ && python setup.py install && cd /work
```

**6. Download Model Weights & Create Dummy Dataset:**
*We download the pretrained DLA34 backbone weights and create an empty `test.txt` file to bypass the dataset loader error for single-image inference.*
```bash
wget https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth
mkdir -p dataset/culane/list
touch dataset/culane/list/test.txt
```

---

### Part 2: Your "Everyday" Workflow
*Since you have already done Part 1 on your laptop, this is all you ever have to do when you sit down to work from now on.*

1. **Start WSL** and go to your folder: `cd ~/VisualComputing/CLRerNet`
2. **Start the container:** `docker-compose run --rm dev`
3. **Run the NMS fix:** `cd libs/models/layers/nms/ && python setup.py install && cd /work`
4. **Run Inference:**
   ```bash
   python demo/image_demo.py demo/demo.jpg configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.png
   ```

---

### Part 3: Why is inference taking 25 seconds?
Look closely at your terminal output:
`Downloading: "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth" ... 00:25`

**The actual AI inference takes less than half a second.** The reason it feels so slow is that the script is spending 25 seconds downloading the base DLA34 neural network weights from the internet every single time you run the command. 

* **Why?** Because Docker containers are temporary (`--rm`), the internal cache where it downloads this file gets deleted when you exit.
* **How to fix it:** The script downloads it to `/home/docker/.cache/torch/hub/checkpoints/`. If you want to make it instant, you can edit your `docker-compose.yaml` and add a new volume under the `volumes:` section to save the cache to your hard drive, like this:
  `- ~/.cache/torch:/home/docker/.cache/torch`

---

### Part 4: Can you run full testing on your RTX 3050?
Since you need to reproduce the paper's results, you will eventually have to evaluate the model on the entire CULane test dataset.

* **Is your hardware enough?** Yes! Testing/Evaluating only processes one image at a time (batch size 1). Your 4GB VRAM and 12GB WSL RAM are perfectly fine for this. *(Note: Training a new model from scratch on 4GB VRAM would crash, but testing is lightweight).*
* **How long will it take?** The CLRerNet with a DLA34 backbone runs at roughly 30 to 40 FPS. The CULane test set contains about **34,680 images**. At 30 FPS, it will take your RTX 3050 roughly **20 to 30 minutes** to run the full evaluation script.

**The Catch (Storage Space):**
To run the full test, you cannot use the dummy `test.txt` file anymore. You will have to download the actual **CULane Dataset**. That dataset is massive—it is over **50GB** in size. Make sure you have enough free space on your D: drive before you start downloading it for the next phase of your project.

Here is the complete, end-to-end markdown summary of the testing process. You can copy this directly into a `.md` file (e.g., `EVALUATION_GUIDE.md`) to include in your Visual Computing project repository.

***

# End-to-End Evaluation Guide: CLRerNet on CULane

This guide outlines the exact steps to reproduce the evaluation of the CLRerNet model (EMA version) on the CULane testing dataset using a Windows/WSL Docker environment. Due to VRAM constraints (e.g., running on a 4GB RTX 3050), this guide strictly covers the **Testing/Evaluation** phase using pre-trained weights, avoiding the massive storage and memory requirements of full model training.

## 1. Dataset Acquisition (Testing Set Only)
To evaluate the model, you do not need the full 50GB CULane dataset. You only need the testing frames and the evaluation list.

Download the following four specific compressed files from the official CULane Google Drive or Kaggle:
1. `driver_37_30frame.tar.gz`
2. `driver_100_30frame.tar.gz`
3. `driver_193_90frame.tar.gz`
4. `list.tar.gz`

Using Windows File Explorer, navigate to the WSL project directory and place these four files into the `dataset/culane` folder:
`\\wsl$\Ubuntu\home\<username>\VisualComputing\CLRerNet\dataset\culane`

## 2. Docker Volume Configuration Fix
By default, the `docker-compose.yaml` file might attempt to map the dataset to a directory outside the project folder. To ensure the Docker container can see the downloaded files, you must fix the volume mapping.

Open `docker-compose.yaml` and **delete** or comment out the following line under the `dev` service `volumes:` section:
```yaml
# Delete this line:
# - $HOME/dataset:/work/dataset
```
Leave the `- .:/work` line intact so the container maps the entire current project directory.

## 3. Data Extraction (WSL Host)
Extracting the Linux `.tar.gz` archives is much faster and less error-prone when done directly in the WSL host terminal rather than inside the Docker container. 

Open a normal WSL terminal and run:

```bash
# Navigate to the dataset folder
cd ~/VisualComputing/CLRerNet/dataset/culane

# Extract the testing images and lists
tar -xzvf driver_37_30frame.tar.gz
tar -xzvf driver_100_30frame.tar.gz
tar -xzvf driver_193_90frame.tar.gz
tar -xzvf list.tar.gz

# Clean up the archives and any Windows identifier files to save space
rm *.tar.gz
rm *Identifier
```

**Folder Structure Verification:**
Ensure the `list/test.txt` file exists. It should contain exactly 34,680 lines.
```bash
wc -l list/test.txt
```

## 4. Running the Evaluation inside Docker
With the dataset in place and the configuration fixed, start the virtual environment and run the evaluation script. Ensure you have the pre-trained weights (`clrernet_culane_dla34_ema.pth`) saved in your main project folder.

```bash
# 1. Start the Docker container
cd ~/VisualComputing/CLRerNet
docker-compose run --rm dev

# 2. Compile the Non-Maximum Suppression (NMS) tool (Required per session)
cd libs/models/layers/nms/ && python setup.py install && cd /work

# 3. Run the MMDetection testing script
python tools/test.py configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth
```

## 5. Expected Results
The evaluation script processes all 34,680 images with a batch size of 64. On an RTX 3050, this takes approximately 25-30 minutes and consumes roughly 3.3 GB of VRAM. 

Upon completion, the script prints a detailed metrics table for three IoU thresholds (0.1, 0.5, and 0.75). 

**Benchmark Target:**
For the standard benchmark threshold ($IoU=0.5$), the official published paper reports an $F1_{50}$ score of **81.43%** for the CLRerNet* model with a DLA34 backbone[cite: 5, 17, 331]. Following these exact steps successfully reproduces an $F1_{50}$ score of **81.55%**, verifying the academic results.

## CLRerNet Reproduction Analysis: Validation vs. Published Baseline

This analysis compares our local reproduction of the CLRerNet model against the official results published in the original paper. 

**Hardware Context:**
* **Host:** Windows Subsystem for Linux (WSL2)
* **GPU:** NVIDIA RTX 3050 Laptop (4GB VRAM)
* **Evaluation Mode:** FP32 Inference, Batch Size 64

### 1. F1 Score Comparison ($IoU = 0.5$)
The following table maps the official paper's metrics for the `CLRerNet†*` (DLA34) model directly against our local evaluation results.

| Evaluation Category | Paper Reported | Local Run | Variance | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 94.36% | 94.36% | `0.00` | Exact Match. |
| **Crowd** | 80.62% | 80.85% | `+0.23` | Slight over-performance. |
| **Dazzle (Highlight)**| 75.23% | 75.17% | `-0.06` | Negligible variance. |
| **Shadow** | 84.35% | 84.55% | `+0.20` | Slight over-performance. |
| **No Line** | 57.31% | 56.75% | `-0.56` | Minor degradation. |
| **Arrow** | 91.17% | 90.99% | `-0.18` | Negligible variance. |
| **Curve** | 79.11% | 78.83% | `-0.28` | Negligible variance. |
| **Night** | 76.92% | 76.85% | `-0.07` | Negligible variance. |
| --- | --- | --- | --- | --- |
| **Total $F1_{50}$** | **81.43%** | **81.55%** | **`+0.12`** | **Successful Reproduction.** |

*Note on "Cross" Category: The Cross category only measures False Positives (FP). The paper reported 1540 FP. Our run achieved 1335 FP, resulting in 205 fewer false positives.*

### 2. Conclusion
The local reproduction achieved a total $F1_{50}$ score of **81.55%**, which falls perfectly within the official paper's stated statistical variance of $81.43 \pm 0.14$. The reproduction is successful, providing a verified baseline to implement and measure novel architectural or algorithmic improvements.


## Proposed Novelty Implementations for CLRerNet

The following methodologies aim to improve the baseline F1 metrics without requiring model retraining. They are specifically designed to execute within a 4GB VRAM constraint.

### Novelty 1: Test-Time Augmentation (TTA) Ensemble
**Target Improvement:** Overall $F1_{50}$ Score
**Concept:** Instead of running the network on a single test image, we run it on multiple augmented versions of the *same* image (e.g., original, horizontally flipped, and slightly scaled). We then aggregate the output coordinates. 
**Why it works:** TTA acts like a poor man's ensemble. If the network misses a lane in the original image but detects it in the flipped version, the aggregated result captures both, drastically reducing False Negatives.
**Hardware Feasibility:** Perfect for 4GB VRAM. It only processes one image at a time, it just does it 3 times per frame. It will lower the FPS, but significantly boost the F1 score.
**Implementation Steps:**
1. Create a custom Python script that imports the `init_detector` and `inference_detector` functions.
2. Read an image, create a flipped copy using OpenCV.
3. Pass both through the network.
4. Un-flip the coordinates of the second output.
5. Apply Non-Maximum Suppression (NMS) on the combined bounding boxes/lines to filter out duplicates.

### Novelty 2: CUDA-Optimized Custom NMS (Non-Maximum Suppression)
**Target Improvement:** FPS / Search Throughput and F1 on `Curve`
**Concept:** The standard NMS filters out overlapping lane predictions using basic intersection metrics. By writing a highly parallelized, curvature-aware NMS algorithm in CUDA C++, we can optimize the exact distance calculations between lane points.
**Why it works:** Replacing standard PyTorch operations with a custom kernel can drastically reduce overhead. Fusing multiple kernels (e.g., distance calculation, sorting, and neighbor filtering) to remove tiny kernel launches per iteration allows the block sizes to better fit GPU occupancy, directly improving throughput. It also prevents the algorithm from accidentally deleting valid curved lanes that happen to intersect at the bottom of the frame.
**Hardware Feasibility:** Excellent. Memory footprint remains identical; purely a computational speedup and logic refinement.
**Implementation Steps:**
1. Navigate to `libs/models/layers/nms/src`.
2. Rewrite the core NMS loop in CUDA C++, implementing an adaptive Bitonic sort or similar fast sorting algorithm for lane confidence scores.
3. Recompile the extension (`python setup.py install`).
4. Rerun the test script to measure the FPS increase and F1 retention.

### Novelty 3: Algorithmic Lane Interpolation for "No Line"
**Target Improvement:** `Noline` Category F1 Score (Currently 56.75%)
**Concept:** Apply geometric heuristics to fix fragmented network outputs. When lane markings are physically absent, the neural network often outputs broken segments.
**Why it works:** We can apply rigorous logic to bridge these gaps. If two distinct lane segments share a similar polynomial trajectory and have a gap of fewer than $X$ pixels, they belong to the same lane.

**Hardware Feasibility:** Zero VRAM cost. This is pure CPU-based algorithmic logic.
**Implementation Steps:**
1. Hook into the output of `inference_detector`.
2. Before the coordinates are formatted for the CULane evaluation metric, run an algorithm that groups lines by their slopes.
3. If the distance between the endpoint of Line A and the start point of Line B is below a threshold, replace both with a single continuous B-spline curve.
4. Evaluate the new coordinates.

### Novelty 4: Image-Entropy Driven Adaptive Thresholding
**Target Improvement:** `Night`, `Highlight`, and `Crowd` Categories
**Concept:** The paper relies on a static confidence threshold (0.43) across the entire dataset. This is suboptimal.
**Why it works:** A dark night image has lower overall activation confidence than a bright daytime image. By calculating the image's overall variance/entropy before inference, we can dynamically scale the confidence threshold. Dark images get a relaxed threshold ($0.35$) to catch faint lanes, while crowded images get a strict threshold ($0.50$) to ignore car edges.
**Hardware Feasibility:** Extremely lightweight. OpenCV calculates image entropy in milliseconds.
**Implementation Steps:**
1. Modify the test loop in MMDetection.
2. Calculate the grayscale standard deviation of the input tensor.
3. Create a linear mapping function: $Threshold = f(StdDev)$.
4. Pass this dynamic threshold into the bounding box filtering stage instead of the static `cfg.test_cfg.conf_threshold`.