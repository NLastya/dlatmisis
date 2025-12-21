import torch
import torchvision
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from transformers import T5EncoderModel, T5Tokenizer, CLIPProcessor, CLIPModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
import open_clip
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from torch import nn
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# CIFAR-10 class names
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck']

# Task 1: Load CIFAR-10 dataset and select sample images
def load_cifar_samples(num_samples=7):
    """Load CIFAR-10 dataset and select sample images from different classes"""
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10 test dataset
    cifar_dataset = CIFAR10(root='./data', train=False, download=True, transform=None)
    
    sample_images = []
    sample_labels = []
    
    # Select one image from each of the first num_samples classes
    for class_idx in range(min(num_samples, len(cifar_classes))):
        # Find first image of this class
        for i, (img, label) in enumerate(cifar_dataset):
            if label == class_idx:
                sample_images.append(img)
                sample_labels.append(label)
                break
    
    print(f"Selected {len(sample_images)} images from {len(set(sample_labels))} different classes")
    return sample_images, sample_labels, cifar_dataset

# Task 2: Create DataFrame with questions and answers
def create_qa_dataframe(sample_images, sample_labels):
    """Create a DataFrame with questions and answers for the sample images"""
    print("Creating QA dataset...")
    
    # Define questions and answers for each image
    qa_data = {
        'image_id': [],
        'question': [],
        'answer': []
    }
    
    # Add questions for each image
    for img_id, label in enumerate(sample_labels):
        class_name = cifar_classes[label]
        
        # Add 2 questions per image
        qa_data['image_id'].append(img_id)
        qa_data['question'].append(f"What is in this image?")
        qa_data['answer'].append(class_name)
        
        qa_data['image_id'].append(img_id)
        qa_data['question'].append(f"Is this a {class_name}?")
        qa_data['answer'].append("yes")
    
    df = pd.DataFrame(qa_data)
    print(f"Created dataset: {len(df)} questions for {len(sample_images)} images")
    print(df.head())
    
    return df

# Task 3: Implement ImageEncoder class with ResNet50
class ImageEncoder:
    def __init__(self):
        print("Initializing ImageEncoder with ResNet50...")
        # Load pretrained ResNet50 and remove the classification layer
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last layer
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def encode(self, images):
        """Encode images to embeddings"""
        print(f"Encoding {len(images)} images...")
        embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Transform image
                img_tensor = self.transform(img).unsqueeze(0).to(device)
                
                # Get embedding
                emb = self.model(img_tensor)
                emb = emb.squeeze().cpu().numpy()
                embeddings.append(emb)
        
        return np.array(embeddings)

# Task 4: Implement TextEncoder class with T5-small
class TextEncoder:
    def __init__(self, model_name='t5-small'):
        print(f"Initializing TextEncoder with {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
    
    def encode(self, texts):
        """Encode texts to embeddings"""
        print(f"Encoding {len(texts)} texts...")
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embedding
                outputs = self.model(**inputs)
                # Use mean pooling over sequence length
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)
        
        return np.array(embeddings)

# Task 5: Implement VQAClassifier with MLP
class VQAClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=512):
        super().__init__()
        print(f"Initializing VQAClassifier with image_dim={image_dim}, text_dim={text_dim}, num_classes={num_classes}")
        
        # MLP layers
        self.fc1 = nn.Linear(image_dim + text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, image_emb, text_emb):
        # Concatenate embeddings
        x = torch.cat([image_emb, text_emb], dim=1)
        
        # Forward through MLP
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Task 6: Train the baseline model
def train_baseline_model(image_embeddings, question_embeddings, df, epochs=10, batch_size=8):
    """Train the baseline VQA model"""
    print("Training baseline model...")
    
    # Create answer vocabulary
    answer_vocab = {ans: idx for idx, ans in enumerate(df['answer'].unique())}
    idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
    
    print(f"Answer vocabulary ({len(answer_vocab)} classes): {list(answer_vocab.keys())}")
    
    # Prepare data - we need to match image embeddings with questions
    # Since we have 2 questions per image, we need to duplicate image embeddings
    image_ids = df['image_id'].values
    X_image = torch.FloatTensor(image_embeddings[image_ids])
    X_text = torch.FloatTensor(question_embeddings)
    y = torch.LongTensor([answer_vocab[ans] for ans in df['answer']])
    
    # Split data
    X_img_train, X_img_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_image, X_text, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = VQAClassifier(
        image_dim=image_embeddings.shape[1],
        text_dim=question_embeddings.shape[1],
        num_classes=len(answer_vocab)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        for i in range(0, len(X_img_train), batch_size):
            batch_img = X_img_train[i:i+batch_size].to(device)
            batch_text = X_text_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_img, batch_text)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / (len(X_img_train) // batch_size)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_img = X_img_test.to(device)
        test_text = X_text_test.to(device)
        test_y = y_test.to(device)
        
        outputs = model(test_img, test_text)
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy = 100 * (predicted == test_y).sum().item() / test_y.size(0)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return model, answer_vocab, idx_to_answer, train_losses, train_accuracies, test_accuracy

# Task 7: Test the baseline model
def predict_baseline(model, image_encoder, text_encoder, image, question, answer_vocab, idx_to_answer):
    """Predict answer using baseline model"""
    model.eval()
    
    with torch.no_grad():
        # Encode image and question
        img_emb = image_encoder.encode([image])
        text_emb = text_encoder.encode([question])
        
        # Convert to tensors
        img_tensor = torch.FloatTensor(img_emb).to(device)
        text_tensor = torch.FloatTensor(text_emb).to(device)
        
        # Get prediction
        outputs = model(img_tensor, text_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        # Convert to answer
        answer_idx = predicted.item()
        return idx_to_answer[answer_idx]

# Task 8: Implement CLIP zero-shot VQA
def load_clip_model():
    """Load CLIP model"""
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model.to(device)
    clip_model.eval()
    return clip_model, clip_processor

def predict_clip(clip_model, clip_processor, image, question, candidate_answers):
    """Predict answer using CLIP zero-shot approach"""
    print(f"Predicting with CLIP for question: {question}")
    
    # Create prompts for each answer
    prompts = [f"Question: {question}. Answer: {answer}" for answer in candidate_answers]
    
    # Process inputs
    inputs = clip_processor(
        text=prompts, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Get best answer
        best_idx = probs.argmax().item()
        return candidate_answers[best_idx]

# Task 9: Load and test LLaVA model
def load_llava_model():
    """Load LLaVA model"""
    print("Loading LLaVA model...")
    
    try:
        # Try loading with 8-bit quantization to save memory
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            quantization_config=quantization_config,
            device_map="auto"
        )
    except:
        # Fallback to regular loading
        print("Failed to load with quantization, trying regular loading...")
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    return llava_model, llava_processor

def predict_llava(llava_model, llava_processor, image, question):
    """Generate answer using LLaVA"""
    print(f"Predicting with LLaVA for question: {question}")
    
    # Create prompt
    prompt = f"USER: <image>\nQuestion: {question}\nASSISTANT:"
    
    # Process inputs
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        output = llava_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
    # Decode response
    answer = llava_processor.decode(output[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[1].strip()
    
    return answer

# Task 10: Compare results from all models
def compare_models(sample_images, df, image_encoder, text_encoder, baseline_model, 
                   answer_vocab, idx_to_answer, clip_model, clip_processor, 
                   llava_model=None, llava_processor=None):
    """Compare results from all models"""
    print("Comparing models...")
    
    results = {
        'Image ID': [],
        'Question': [],
        'True Answer': [],
        'ResNet+T5': [],
        'CLIP': [],
        'LLaVA': []
    }
    
    # Test on a subset of examples
    test_examples = df.head(min(10, len(df)))
    
    for i, row in test_examples.iterrows():
        image = sample_images[row['image_id']]
        question = row['question']
        true_answer = row['answer']
        
        # Baseline prediction
        baseline_pred = predict_baseline(
            baseline_model, image_encoder, text_encoder, 
            image, question, answer_vocab, idx_to_answer
        )
        
        # CLIP prediction
        clip_pred = predict_clip(
            clip_model, clip_processor, image, question, 
            list(answer_vocab.keys())
        )
        
        # LLaVA prediction (if available)
        llava_pred = "N/A"
        if llava_model is not None and llava_processor is not None:
            try:
                llava_pred = predict_llava(llava_model, llava_processor, image, question)
            except Exception as e:
                print(f"LLaVA prediction failed: {e}")
                llava_pred = "Error"
        
        # Store results
        results['Image ID'].append(row['image_id'])
        results['Question'].append(question)
        results['True Answer'].append(true_answer)
        results['ResNet+T5'].append(baseline_pred)
        results['CLIP'].append(clip_pred)
        results['LLaVA'].append(llava_pred)
    
    results_df = pd.DataFrame(results)
    return results_df

# Task 11: Analyze and visualize results
def calculate_accuracy(predictions, true_answers):
    """Calculate accuracy of predictions"""
    correct = sum(1 for p, t in zip(predictions, true_answers) if p.lower() == t.lower())
    return correct / len(predictions)

def analyze_results(results_df):
    """Analyze and visualize model comparison results"""
    print("\nAnalyzing results...")
    
    # Calculate accuracies
    baseline_acc = calculate_accuracy(results_df['ResNet+T5'], results_df['True Answer'])
    clip_acc = calculate_accuracy(results_df['CLIP'], results_df['True Answer'])
    
    # Calculate LLaVA accuracy (skip N/A and Error values)
    llava_preds = [p for p in results_df['LLaVA'] if p not in ['N/A', 'Error']]
    llava_true = [results_df['True Answer'].iloc[i] for i, p in enumerate(results_df['LLaVA']) 
                  if p not in ['N/A', 'Error']]
    llava_acc = calculate_accuracy(llava_preds, llava_true) if llava_preds else 0
    
    print("\nModel Accuracies:")
    print(f"ResNet+T5: {baseline_acc:.2%}")
    print(f"CLIP: {clip_acc:.2%}")
    print(f"LLaVA: {llava_acc:.2%}")
    
    # Create visualization
    models = ['ResNet+T5', 'CLIP', 'LLaVA']
    accuracies = [baseline_acc, clip_acc, llava_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    return baseline_acc, clip_acc, llava_acc

def visualize_samples(images, df, n_samples=3):
    """Visualize sample images with questions"""
    fig, axes = plt.subplots(1, min(n_samples, len(images)), figsize=(15, 5))
    if n_samples == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            ax.imshow(images[idx])
            ax.axis('off')
            questions = df[df['image_id'] == idx]
            title = f"Image {idx}\n"
            for _, row in questions.iterrows():
                title += f"Q: {row['question'][:30]}...\n"
            ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

# Main function to run all tasks
def main():
    """Main function to execute all VQA tasks"""
    print("=" * 50)
    print("Visual Question Answering (VQA) Solution")
    print("=" * 50)
    
    # Task 1: Load CIFAR-10 samples
    sample_images, sample_labels, cifar_dataset = load_cifar_samples(num_samples=7)
    
    # Task 2: Create QA DataFrame
    df = create_qa_dataframe(sample_images, sample_labels)
    
    # Visualize samples
    visualize_samples(sample_images, df, n_samples=3)
    
    # Task 3: Initialize ImageEncoder
    image_encoder = ImageEncoder()
    image_embeddings = image_encoder.encode(sample_images)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Task 4: Initialize TextEncoder
    text_encoder = TextEncoder()
    question_embeddings = text_encoder.encode(df['question'].tolist())
    print(f"Question embeddings shape: {question_embeddings.shape}")
    
    # Task 5-6: Train baseline model
    baseline_model, answer_vocab, idx_to_answer, train_losses, train_accuracies, test_accuracy = train_baseline_model(
        image_embeddings, question_embeddings, df, epochs=10, batch_size=4
    )
    
    # Task 7: Test baseline model
    print("\nTesting baseline model on examples:")
    for i in range(min(3, len(df))):
        img_id = df.iloc[i]['image_id']
        question = df.iloc[i]['question']
        true_answer = df.iloc[i]['answer']
        
        predicted = predict_baseline(
            baseline_model, image_encoder, text_encoder,
            sample_images[img_id], question, answer_vocab, idx_to_answer
        )
        
        print(f"Image {img_id}, Q: {question}")
        print(f"  True: {true_answer}, Predicted: {predicted}")
        print()
    
    # Task 8: Load CLIP model
    clip_model, clip_processor = load_clip_model()
    
    # Test CLIP on examples
    print("\nTesting CLIP on examples:")
    for i in range(min(3, len(df))):
        img_id = df.iloc[i]['image_id']
        question = df.iloc[i]['question']
        true_answer = df.iloc[i]['answer']
        
        predicted = predict_clip(
            clip_model, clip_processor, sample_images[img_id], 
            question, list(answer_vocab.keys())
        )
        
        print(f"Image {img_id}, Q: {question}")
        print(f"  True: {true_answer}, Predicted: {predicted}")
        print()
    
    # Task 9: Try to load LLaVA (optional, may fail due to memory constraints)
    llava_model = None
    llava_processor = None
    try:
        llava_model, llava_processor = load_llava_model()
        
        # Test LLaVA on one example
        print("\nTesting LLaVA on example:")
        img_id = df.iloc[0]['image_id']
        question = df.iloc[0]['question']
        true_answer = df.iloc[0]['answer']
        
        predicted = predict_llava(
            llava_model, llava_processor, sample_images[img_id], question
        )
        
        print(f"Image {img_id}, Q: {question}")
        print(f"  True: {true_answer}, Predicted: {predicted}")
    except Exception as e:
        print(f"Failed to load or run LLaVA: {e}")
        print("Continuing without LLaVA...")
    
    # Task 10: Compare models
    results_df = compare_models(
        sample_images, df, image_encoder, text_encoder, baseline_model,
        answer_vocab, idx_to_answer, clip_model, clip_processor,
        llava_model, llava_processor
    )
    
    print("\nModel Comparison Results:")
    print(results_df)
    
    # Task 11: Analyze results
    baseline_acc, clip_acc, llava_acc = analyze_results(results_df)
    
    # Print conclusions
    print("\n" + "=" * 50)
    print("CONCLUSIONS")
    print("=" * 50)
    
    print("\nBaseline (ResNet + T5):")
    print("- Strong sides: Simple architecture, fast training, interpretable")
    print("- Weak sides: Limited by small dataset, may not generalize well")
    
    print("\nCLIP:")
    print("- Strong sides: Zero-shot capability, no training required, good performance")
    print("- Weak sides: Limited to predefined answer choices, may struggle with complex questions")
    
    print("\nLLaVA:")
    print("- Strong sides: State-of-the-art performance, can generate free-form answers")
    print("- Weak sides: Requires significant GPU memory, slower inference")
    
    print("\nGeneral observations:")
    print(f"- Baseline achieved {baseline_acc:.2%} accuracy on test set")
    print(f"- CLIP achieved {clip_acc:.2%} accuracy without any training")
    if llava_acc > 0:
        print(f"- LLaVA achieved {llava_acc:.2%} accuracy")
    else:
        print("- LLaVA was not successfully tested due to resource constraints")
    
    print("\nFor practical applications with limited resources, CLIP provides the best balance")
    print("of performance and efficiency. For research or high-performance applications,")
    print("LLaVA or similar large multimodal models would be preferred.")

if __name__ == "__main__":
    main()
