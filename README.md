# 💠 Find the Crystals
A Mini Proof-of-Concept Protein Crystallization Image Classifier using GPT4.  
🏆 Runner-up in [London, Paris, Berlin AI HackXelerator™ - LPB25](https://www.kxsb.org/lpb25#winners) Imaging track!

## Demo
Try for yourself!
[Find the Crystals](https://huggingface.co/spaces/asphodel-thuang/microscopy-vlm-lpb25)


## 🔷 Why Protein Crystals:
Proteins are the tiny machines that keep all living things running, and understanding their shape is key to understanding how they work — and how to design new medicines.
One powerful way to figure out a protein's 3D structure is by turning it into a crystal and analyzing it with X-ray crystallography. But here's the catch: proteins don't like to crystallize. 
Scientists often need to run thousands of experiments just to grow one usable crystal.  
To this end, MARCO (MAchine Recognition of Crystallization Outcomes) was established to bring real-world crystallization data to the machine learning community.
The goal is to build smart tools that can automatically classify the outcomes of these tricky experiments. 
By helping scientists spot crystals faster and more accurately, these tools can accelerate discoveries in biology and medicine.


## 🔮 How Does It Work
### Prompt Design  
As you've noticed, this classifier is not a custom-trained CNN image classifier.
Instead, it uses GPT-4, a Large Language Model with Vision and Reasoning capability, to classify images based on a prompt.
Sepecifically, the prompt covers the following parts:
- **Instructions:** A short task instruction with domain context and category definitions
- **Examples:** 1 image from each of the 4 categories are provided as examples, along with an explanation to facilitate reasoning
- **Test Image:** The actual test image, different from the example images, to be classified

### Batch Accuracy
🔶 **Note that this is only a very priliminary proof of concept.** 🔶  
The current prompt was tested on a batch of **79** images (~20 each category), from a single source of image provider, with the following results:
- **Accuracy:** 75%
- **Precision:** 80%
- **Recall:** 75%
- **F1-score:** 75%
- **Classification Report:**

| category | precision | recall | f1-score | sample size |
|---|---|---|---|---|
| clear | 0.79 | 0.95 | 0.86 | 20 |
| crystals | 0.93 | 0.68 | 0.79 | 19 |
| other | 0.92 | 0.55 | 0.69 | 20 |
| precipitate | 0.55 | 0.80 | 0.65 | 20 |
| average | 0.80 | 0.75 | 0.75 | 79 |

## 📖 Reference:
1. Dataset & Background: [MARCO](https://marco.ccr.buffalo.edu/about).

## How to Run

1. Prepare environment:
```bash
python3.10 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

2. Check out the `research` folder for experiment notebooks.  
Note that these are for quick prototyping not well tested tutorials.
