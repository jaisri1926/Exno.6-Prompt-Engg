# Exno.6-Prompt-Engg
## Register no.212222060086
## Aim: 
    Development of Python Code Compatible with Multiple AI Tools
# Algorithm: 
     Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.
# **AI-Powered API Integration & Analysis Tool**

Below is a Python implementation that integrates with multiple AI tools (OpenAI, Hugging Face, and Google Gemini) to:
1. Automate API interactions
2. Compare outputs from different AI models
3. Generate actionable insights
python
```
import os
import json
import requests
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient

class AIToolComparator:
    def __init__(self):
        # Initialize all AI clients with API keys
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.hf_client = InferenceClient(token=os.getenv('HF_API_KEY'))
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_client = genai.GenerativeModel('gemini-pro')
        
        # Store results for comparison
        self.results = {
            'query': [],
            'openai_response': [],
            'hf_response': [],
            'gemini_response': []
        }

    def query_openai(self, prompt):
        """Query OpenAI's GPT model"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def query_huggingface(self, prompt):
        """Query Hugging Face's inference API"""
        response = self.hf_client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.1"
        )
        return response

    def query_gemini(self, prompt):
        """Query Google's Gemini model"""
        response = self.gemini_client.generate_content(prompt)
        return response.text

    def run_comparison(self, prompts):
        """Run comparison across all AI tools for given prompts"""
        for prompt in prompts:
            self.results['query'].append(prompt)
            
            try:
                self.results['openai_response'].append(self.query_openai(prompt))
            except Exception as e:
                self.results['openai_response'].append(f"Error: {str(e)}")
            
            try:
                self.results['hf_response'].append(self.query_huggingface(prompt))
            except Exception as e:
                self.results['hf_response'].append(f"Error: {str(e)}")
            
            try:
                self.results['gemini_response'].append(self.query_gemini(prompt))
            except Exception as e:
                self.results['gemini_response'].append(f"Error: {str(e)}")

    def generate_insights(self):
        """Generate comparative insights from the results"""
        df = pd.DataFrame(self.results)
        
        # Calculate response lengths
        df['openai_length'] = df['openai_response'].apply(len)
        df['hf_length'] = df['hf_response'].apply(len)
        df['gemini_length'] = df['gemini_response'].apply(len)
        
        # Generate basic comparison
        insights = {
            'average_response_lengths': {
                'OpenAI': df['openai_length'].mean(),
                'HuggingFace': df['hf_length'].mean(),
                'Gemini': df['gemini_length'].mean()
            },
            'response_variability': "High" if len(set(df['openai_response'] + df['hf_response'] + df['gemini_response'])) > len(df)*0.8 else "Medium/Low"
        }
        
        return df, insights

    def save_results(self, filename="ai_comparison_results"):
        """Save results to JSON and CSV"""
        df, insights = self.generate_insights()
        
        # Save to JSON
        with open(f"{filename}.json", 'w') as f:
            json.dump({
                'raw_results': self.results,
                'insights': insights
            }, f, indent=2)
        
        # Save to CSV
        df.to_csv(f"{filename}.csv", index=False)
        
        return df, insights


# Example Usage
if __name__ == "__main__":
    # Load API keys from environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize comparator
    comparator = AIToolComparator()
    
    # Define test prompts
    test_prompts = [
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate Fibonacci sequence",
        "What are the main differences between React and Vue.js?"
    ]
 
    # Run comparison
    comparator.run_comparison(test_prompts)
    
    # Generate and save results
    results_df, insights = comparator.save_results()
    
    print("\nComparative Insights:")
    print(json.dumps(insights, indent=2))
    
    print("\nSample Results:")
    print(results_df.head())
```

## **Key Features**

1. **Multi-AI Integration**:
   - OpenAI GPT-3.5
   - Hugging Face (Mistral-7B)
   - Google Gemini

2. **Automated Comparison**:
   - Queries all APIs with the same prompt
   - Stores responses in structured format
   - Handles API errors gracefully

3. **Insight Generation**:
   - Calculates response lengths
   - Measures response variability
   - Provides quantitative comparisons

4. **Data Export**:
   - Saves results to JSON and CSV
   - Preserves raw responses and insights

## **How to Use**

1. Install requirements:
   ```bash
   pip install openai huggingface-hub google-generativeai pandas python-dotenv requests
   ```

2. Create `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key
   HF_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```

3. Run the script:
   ```bash
   python ai_comparator.py
   ```

## **Potential Enhancements**

1. Add more AI services (Anthropic, Cohere, etc.)
2. Implement response quality scoring
3. Add latency measurements
4. Create visualization of results
5. Add unit testing

This tool provides a foundation for systematically comparing different AI services and can be extended for more sophisticated analysis as needed.




# Result: 
    The corresponding Prompt is executed successfully
