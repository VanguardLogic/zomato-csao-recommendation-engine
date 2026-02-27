# src/data_preprocessing/generate_synthetic_data.py

import pandas as pd
import random
import os
from datetime import datetime, timedelta

def generate_orders(num_orders=15000):
    dish_pool = {
        "North Indian": [
            "Butter Chicken", "Dal Makhani", "Matar Paneer", "Chole Bhature", "Rajma Chawal",
            "Paneer Tikka", "Garlic Naan", "Butter Naan", "Tandoori Roti", "Palak Paneer", 
            "Kadai Paneer", "Shahi Paneer", "Malai Kofta", "Dal Tadka", "Jeera Rice", 
            "Peas Pulao", "Missi Roti", "Pudina Paratha", "Aloo Paratha", "Gobi Paratha", 
            "Mix Veg", "Bhindi Masala", "Baingan Bharta", "Dum Aloo", "Chicken Curry", 
            "Mutton Rogan Josh", "Keema Naan", "Chicken Tikka", "Seekh Kebab", "Fish Tikka", 
            "Tandoori Chicken", "Afghani Chicken", "Mutton Korma", "Chicken Korma", "Paneer Butter Masala",
            "Chopped Onions", "Papad", "Sirka Pyaaz", "Laccha Pyaaz", "Achaar", "Mint Chutney"
        ],
        "South Indian": [
            "Masala Dosa", "Idli", "Hyderabadi Biryani", "Chicken Chettinad", "Parotta", 
            "Appam", "Coconut Chutney", "Sambar", "Dosa", "Medu Vada", "Mirchi Ka Salan",
            "Rawa Dosa", "Mysore Masala Dosa", "Onion Uthappam", "Tomato Uthappam", "Paneer Dosa",
            "Cheese Dosa", "Podi Idli", "Mini Idli", "Vada", "Rasam", "Lemon Rice",
            "Curd Rice", "Tamarind Rice", "Vangi Bath", "Bisi Bele Bath", "Chicken 65",
            "Guntur Chicken", "Andhra Chicken Curry", "Meen Moilee", "Prawn Roast",
            "Mutton Chukka", "Kothu Parotta", "Idiyappam", "Malabar Parotta", "Veg Stew"
        ],
        "Indo-Chinese": [
            "Momos", "Paneer Momos", "Hakka Noodles", "Manchow Soup", "Clear Soup", 
            "Spring Rolls", "Fried Rice", "Chilli Paneer", "Veg Manchurian", "Chicken Manchurian",
            "Chilli Chicken", "Garlic Chicken", "Schezwan Noodles", "Schezwan Fried Rice",
            "Triple Schezwan Rice", "American Chopsuey", "Chinese Bhel", "Crispy Veg",
            "Honey Chilli Potatoes", "Dragon Chicken", "Sweet Corn Soup", "Hot and Sour Soup",
            "Lemon Coriander Soup", "Taluman Soup", "Crispy Noodles", "Red Chutney", "Mayonnaise"
        ],
        "Fast Food": [
            "Burger", "Hot Dog", "Fries", "Onion Rings", "Extra Cheese Dip", "Mustard",
            "Veg Burger", "Chicken Burger", "Cheese Burger", "Double Patty Burger",
            "Peri Peri Fries", "Cheesy Fries", "Potato Wedges", "Chicken Nuggets",
            "Chicken Wings", "Fish and Chips", "Veg Wrap", "Chicken Wrap", "Paneer Tikka Roll",
            "Egg Roll", "Chicken Roll", "Mutton Roll", "Shawarma", "Falafel", "Pita Bread",
            "Tahini", "Garlic Sauce", "Ketchup"
        ],
        "Italian": [
            "Spaghetti", "Pasta", "Pizza", "Garlic Bread", "Cheese Dip", "Margherita Pizza",
            "Pepperoni Pizza", "Veggie Supreme Pizza", "Farmhouse Pizza", "Penne Arrabbiata",
            "Alfredo Pasta", "Mac and Cheese", "Lasagna", "Ravioli", "Risotto", "Bruschetta",
            "Garlic Breadsticks", "Stuffed Garlic Bread", "Cheese Burst Pizza", "Thin Crust Pizza"
        ],
        "Desserts": [
            "Gulab Jamun", "Rosogolla", "Mishti Doi", "Double Ka Meetha", "Brownie",
            "Rasmalai", "Jalebi", "Rabri", "Kaju Katli", "Barfi", "Ladoo", "Moong Dal Halwa",
            "Gajar Ka Halwa", "Ice Cream", "Chocolate Cake", "Cheesecake", "Tiramisu",
            "Waffles", "Pancakes", "Churros", "Donut", "Muffin", "Fruit Salad", "Kulfi",
            "Falooda", "Shahi Tukda", "Petha", "Soan Papdi", "Mysore Pak"
        ],
        "Beverages": [
            "Water", "Soft Drink", "Coke", "Masala Soda", "Lassi", "Buttermilk", "Chaas",
            "Fruit Beer", "Sweet Lassi", "Nimbu Pani", "Filter Coffee", "Masala Chai",
            "Cold Coffee", "Iced Tea", "Lemonade", "Mojito", "Virgin Mojito", "Blue Lagoon",
            "Fresh Lime Soda", "Mango Shake", "Banana Shake", "Strawberry Shake", "Chocolate Shake",
            "Oreo Shake", "Kitkat Shake", "Cold Drink", "Diet Coke", "Sprite", "Fanta",
            "Thums Up", "Limca", "Ginger Ale", "Tonic Water", "Sparkling Water", "Coconut Water",
            "Sugarcane Juice", "Watermelon Juice", "Orange Juice", "Mosambi Juice", "Apple Juice"
        ]
    }
    
    def get_price(item):
        val = abs(hash(item)) % 300
        if item in dish_pool["Beverages"]: return 20 + (val % 80)
        if item in dish_pool["Desserts"]: return 50 + (val % 100)
        if item in dish_pool["Fast Food"]: return 100 + (val % 150)
        return 100 + (val % 250)

    all_items = []
    for cat in dish_pool: all_items.extend(dish_pool[cat])

    user_profiles = [{"uid": f"U_{i}", "vr": round(random.uniform(0, 1), 2)} for i in range(100)]
    data = []
    
    for _ in range(num_orders):
        region = random.choice(list(dish_pool.keys()))
        user = random.choice(user_profiles)
        main_dish = random.choice(dish_pool[region])
        
        candidates = list(set(dish_pool[region] + dish_pool["Beverages"] + dish_pool["Desserts"]))
        random.shuffle(candidates)
        candidates = candidates[:50]
        
        for addon in candidates:
            if addon == main_dish: continue
            
            # Feature: is_veg
            is_veg = 1
            non_veg_keywords = ["Chicken", "Mutton", "Fish", "Prawn", "Keema", "Meat", "Pepperoni", "Egg"]
            if any(kw in addon for kw in non_veg_keywords): is_veg = 0
            
            label = 0
            if addon in dish_pool[region] and random.random() > 0.85: label = 1
            if ("Pizza" in main_dish and "Garlic" in addon) or ("Burger" in main_dish and "Fries" in addon):
                if random.random() > 0.2: label = 1
            
            data.append({
                "order_id": f"ORD_{_}", "timestamp": None, "user_id": user["uid"],
                "user_historical_veg_ratio": user["vr"], "user_segment": random.choice(["Budget", "Premium"]),
                "order_frequency": "Medium", "time_of_day": random.choice(["Lunch", "Dinner"]),
                "region": region, "cart_items": main_dish, "cart_total_value": get_price(main_dish),
                "candidate_item": addon, "addon_price": get_price(addon), 
                "is_veg": is_veg, "added": label
            })
            
    df = pd.DataFrame(data)
    df['timestamp'] = datetime.now()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_orders.csv", index=False)
    # Split for temporal consistency
    split = int(len(df) * 0.8)
    df.iloc[:split].to_csv("data/historical_train_data.csv", index=False)
    df.iloc[split:].to_csv("data/recent_test_data.csv", index=False)
    print("SUCCESS: Synthetic data generated.")
    return df

if __name__ == "__main__":
    generate_orders()
