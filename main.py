#!/usr/bin/env python3
"""
Fake News Detection - Main User Interface

This script provides a command-line interface for detecting fake news articles.
Users can input news articles and get predictions on whether they are fake or real.
"""

import os
import sys
from src.detector import FakeNewsDetector


def main():
    """Main function to run the fake news detection interface."""
    
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTOR")
    print("=" * 60)
    print("Welcome to the Fake News Detection System!")
    print("This tool analyzes news articles to determine if they are fake or real.")
    print()
    
    # Define paths to saved model and vectorizer
    model_path = "models/fake_news_model.h5"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model first using the training notebook.")
        print("Run: jupyter notebook notebooks/02_model_training.ipynb")
        return
    
    if not os.path.exists(vectorizer_path):
        print(f"❌ Error: Vectorizer file not found at {vectorizer_path}")
        print("Please train the model first using the training notebook.")
        print("Run: jupyter notebook notebooks/02_model_training.ipynb")
        return
    
    try:
        # Initialize the detector
        print("🔄 Loading the fake news detection model...")
        detector = FakeNewsDetector(model_path, vectorizer_path)
        print("✅ Model loaded successfully!")
        print()
        
        # Main interaction loop
        while True:
            print("-" * 60)
            print("📰 Enter a news article to analyze:")
            print("(Type 'quit' or 'exit' to stop)")
            print("-" * 60)
            
            # Get user input
            user_input = input("> ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Fake News Detector!")
                break
            
            # Check for empty input
            if not user_input:
                print("⚠️  Please enter some text to analyze.")
                continue
            
            # Check for help command
            if user_input.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            # Analyze the text
            try:
                print("\n🔄 Analyzing the article...")
                prediction, confidence = detector.predict_with_confidence(user_input)
                
                # Display results
                print("\n" + "=" * 40)
                print("📊 ANALYSIS RESULTS")
                print("=" * 40)
                
                if prediction == "Fake News":
                    print(f"🚨 Prediction: {prediction}")
                    print(f"📈 Confidence: {confidence:.2%}")
                    print("⚠️  This article appears to be fake news.")
                elif prediction == "Real News":
                    print(f"✅ Prediction: {prediction}")
                    print(f"📈 Confidence: {confidence:.2%}")
                    print("✅ This article appears to be legitimate news.")
                else:
                    print(f"❌ {prediction}")
                
                print("=" * 40)
                
            except Exception as e:
                print(f"❌ Error analyzing the text: {str(e)}")
                print("Please try again with different text.")
            
            print()
    
    except Exception as e:
        print(f"❌ Error initializing the detector: {str(e)}")
        print("Please check that the model files are properly saved.")


def print_help():
    """Print help information for the user."""
    print("\n" + "=" * 50)
    print("📖 HELP - FAKE NEWS DETECTOR")
    print("=" * 50)
    print("This tool analyzes news articles to detect fake news.")
    print()
    print("Commands:")
    print("  • Enter any news article text to analyze it")
    print("  • Type 'quit', 'exit', or 'q' to stop the program")
    print("  • Type 'help', 'h', or '?' to show this help")
    print()
    print("Tips:")
    print("  • Enter complete articles for better accuracy")
    print("  • The tool works best with news articles in English")
    print("  • Results include confidence scores")
    print("=" * 50)


if __name__ == "__main__":
    main()
