import pandas as pd
import os

# Existing base data
data = [
    ("it is not good and the product is damaged", "negative"),
    ("the quality is poor and I am not happy", "negative"),
    ("not worth the price at all", "negative"),
    ("waste of money, do not buy", "negative"),
    ("terrible quality, very disappointed", "negative"),
    ("absolutely fantastic, exceeded expectations", "positive"),
    ("great product, highly recommended", "positive"),
    ("excellent value for money", "positive"),
    ("best purchase I have made", "positive"),
    ("I love the design and quality", "positive"),
    ("the product is okay, nothing special", "neutral"),
    ("it is decent but could be better", "neutral"),
    ("neutral opinion, it does the job", "neutral"),
    ("average quality considering the price", "neutral"),
]

# Load original data if exists
if os.path.exists("Datasets_Merged.csv"):
    original_df = pd.read_csv("Datasets_Merged.csv")
    new_data = pd.DataFrame(data, columns=['review', 'sentiment'])
    
    # We will "Boost" the negative negation examples by repeating them 
    # to make sure the model pays attention
    negations_boost = [
        ("it is not good", "negative"),
        ("is not good", "negative"),
        ("not good", "negative"),
        ("not great", "negative"),
        ("not happy", "negative"),
        ("never buy again", "negative"),
        ("no good", "negative"),
        ("not working", "negative"),
        ("damaged product", "negative"),
        ("quality is bad", "negative"),
        ("poor quality", "negative"),
        ("item is broken", "negative"),
        ("totally useless", "negative"),
        ("horrible experience", "negative"),
        ("worst product", "negative"),
        ("dont waste your time", "negative"),
        ("regret buying this", "negative"),
        ("avoid like the plague", "negative"),
        ("very poor", "negative")
    ] * 20
    
    neutral_boost = [
        ("it is okay, nothing special", "neutral"),
        ("the design is good but performance is just fine", "neutral"),
        ("average quality considering the price", "neutral"),
        ("neutral opinion, it does the job", "neutral"),
        ("it is decent but could be better", "neutral"),
        ("just fine", "neutral"),
        ("good but fine", "neutral"),
        ("okay for now", "neutral"),
        ("standard functionality", "neutral"),
        ("neither good nor bad", "neutral"),
        ("not great but not bad either", "neutral"),
        ("not bad, not good", "neutral"),
        ("fairly decent", "neutral"),
        ("performs as expected but lacks wow factor", "neutral"),
        ("it is an average product", "neutral"),
        ("fair price for fair quality", "neutral"),
        ("could be improved", "neutral"),
        ("mediocre experience", "neutral"),
        ("met expectations but nothing more", "neutral"),
        ("it's a product", "neutral"),
        ("not bad", "neutral"),
        ("could be better", "neutral"),
        ("not what I expected but not worse", "neutral")
    ] * 25

    positive_boost = [
        ("absolutely fantastic", "positive"),
        ("great product", "positive"),
        ("highly recommended", "positive"),
        ("excellent value", "positive"),
        ("love it", "positive"),
        ("best purchase", "positive"),
        ("perfect quality", "positive"),
        ("superb performance", "positive"),
        ("very happy with it", "positive"),
        ("five stars", "positive"),
        ("stunning design", "positive"),
        ("fast delivery and great item", "positive"),
        ("quite good", "positive"),
        ("really impressed", "positive"),
        ("better than expected", "positive"),
        ("amazing results", "positive"),
        ("top notch", "positive"),
        ("brilliant idea", "positive")
    ] * 20
    
    boost_df = pd.DataFrame(negations_boost + neutral_boost + positive_boost, columns=['review', 'sentiment'])
    
    # Combine everything
    optimized_df = pd.concat([original_df, new_data, boost_df], ignore_index=True)
    
    # Save as optimized
    optimized_df.to_csv("Datasets_Merged.csv", index=False)
    print("Optimized dataset created as Datasets_Merged.csv")
else:
    # Create from scratch if somehow deleted
    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    df.to_csv("Datasets_Merged.csv", index=False)
    print("New base dataset created as Datasets_Merged.csv")
