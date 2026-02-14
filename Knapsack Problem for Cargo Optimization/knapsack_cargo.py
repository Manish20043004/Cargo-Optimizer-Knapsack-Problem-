import time
import random
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

class CargoItem:
    """Represents an individual cargo item with weight, volume, and value."""
    
    def __init__(self, name, weight, volume, value):
        self.name = name
        self.weight = weight  # in kg
        self.volume = volume  # in cubic meters
        self.value = value    # priority value (higher = more important)
        
    def __repr__(self):
        return f"{self.name} (W:{self.weight}kg, V:{self.volume}m³, Value:{self.value})"

class KnapsackSolver:
    """Class implementing different algorithms to solve the Knapsack problem."""
    
    @staticmethod
    def greedy_solution(items, weight_capacity):
        """Greedy approach based on value/weight ratio."""
        # Calculate value/weight ratio for each item
        for item in items:
            item.ratio = item.value / item.weight
            
        # Sort items by value/weight ratio in descending order
        sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
        
        selected_items = []
        total_weight = 0
        total_value = 0
        
        for item in sorted_items:
            if total_weight + item.weight <= weight_capacity:
                selected_items.append(item)
                total_weight += item.weight
                total_value += item.value
                
        return {
            "selected_items": selected_items,
            "total_weight": total_weight,
            "total_value": total_value,
            "algorithm": "Greedy"
        }
    
    @staticmethod
    def dynamic_programming_solution(items, weight_capacity):
        """Dynamic Programming solution for the 0-1 Knapsack Problem."""
        n = len(items)
        
        # Initialize DP table with zeros
        dp = [[0 for _ in range(weight_capacity + 1)] for _ in range(n + 1)]
        
        # Fill the DP table
        for i in range(1, n + 1):
            for w in range(weight_capacity + 1):
                if items[i-1].weight <= w:
                    dp[i][w] = max(
                        items[i-1].value + dp[i-1][w - int(items[i-1].weight)],
                        dp[i-1][w]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtrack to find the selected items
        selected_items = []
        total_value = dp[n][weight_capacity]
        w = weight_capacity
        
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_items.append(items[i-1])
                w -= int(items[i-1].weight)
        
        total_weight = sum(item.weight for item in selected_items)
        
        return {
            "selected_items": selected_items,
            "total_weight": total_weight,
            "total_value": total_value,
            "algorithm": "Dynamic Programming"
        }
        
    @staticmethod
    def branch_and_bound_solution(items, weight_capacity):
        """Branch and Bound solution for the 0-1 Knapsack Problem."""
        # Sort items by value/weight ratio in descending order
        for item in items:
            item.ratio = item.value / item.weight
        sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
        
        # Node structure for branch and bound
        class Node:
            def __init__(self, level, profit, weight, bound, includes):
                self.level = level      # Level in the decision tree
                self.profit = profit    # Profit so far
                self.weight = weight    # Weight so far
                self.bound = bound      # Upper bound on profit
                self.includes = includes  # List tracking included items
                
        # Calculate upper bound for a node
        def bound(node, n, W, items):
            if node.weight >= W:
                return 0
                
            profit_bound = node.profit
            j = node.level + 1
            totweight = node.weight
            
            # Include items with the highest value/weight ratio
            while j < n and totweight + items[j].weight <= W:
                totweight += items[j].weight
                profit_bound += items[j].value
                j += 1
                
            # Include fractional part of the next item
            if j < n:
                profit_bound += (W - totweight) * items[j].ratio
                
            return profit_bound
        
        n = len(sorted_items)
        max_profit = 0
        best_solution = []
        
        # Create root node
        root = Node(-1, 0, 0, 0, [False] * n)
        root.bound = bound(root, n, weight_capacity, sorted_items)
        
        # Initialize queue with root node
        queue = [root]
        
        while queue:
            # Extract node with highest bound
            node = max(queue, key=lambda x: x.bound)
            queue.remove(node)
            
            # Skip if bound is less than current best solution
            if node.bound < max_profit:
                continue
                
            # Consider including the next item
            level = node.level + 1
            if level < n:
                # Copy the included items list
                includes_yes = node.includes.copy()
                includes_yes[level] = True
                
                # Check if we can include this item
                if node.weight + sorted_items[level].weight <= weight_capacity:
                    yes_node = Node(
                        level,
                        node.profit + sorted_items[level].value,
                        node.weight + sorted_items[level].weight,
                        0,
                        includes_yes
                    )
                    yes_node.bound = bound(yes_node, n, weight_capacity, sorted_items)
                    
                    if yes_node.profit > max_profit and yes_node.weight <= weight_capacity:
                        max_profit = yes_node.profit
                        best_solution = includes_yes.copy()
                    
                    if yes_node.bound > max_profit:
                        queue.append(yes_node)
                
                # Consider excluding the next item
                includes_no = node.includes.copy()
                includes_no[level] = False
                no_node = Node(level, node.profit, node.weight, 0, includes_no)
                no_node.bound = bound(no_node, n, weight_capacity, sorted_items)
                
                if no_node.bound > max_profit:
                    queue.append(no_node)
        
        # Get selected items
        selected_items = [sorted_items[i] for i in range(n) if best_solution[i]]
        total_weight = sum(item.weight for item in selected_items)
        
        return {
            "selected_items": selected_items,
            "total_weight": total_weight,
            "total_value": max_profit,
            "algorithm": "Branch and Bound"
        }

def generate_random_cargo_items(n, weight_range=(5, 100), volume_range=(0.1, 5), value_range=(50, 1000)):
    """Generate n random cargo items for testing."""
    cargo_types = [
        "Electronics", "Textiles", "Automotive Parts", "Food Products", 
        "Pharmaceuticals", "Machine Parts", "Consumer Goods", "Raw Materials",
        "Medical Supplies", "Luxury Goods", "Chemical Products", "Construction Materials"
    ]
    
    items = []
    for i in range(n):
        cargo_type = random.choice(cargo_types)
        name = f"{cargo_type}-{i+1}"
        weight = round(random.uniform(*weight_range), 1)
        volume = round(random.uniform(*volume_range), 2)
        value = round(random.uniform(*value_range))
        
        items.append(CargoItem(name, weight, volume, value))
    
    return items

def compare_algorithms(items, weight_capacity):
    """Compare different algorithms for solving the knapsack problem."""
    algorithms = [
        KnapsackSolver.greedy_solution,
        KnapsackSolver.dynamic_programming_solution,
        KnapsackSolver.branch_and_bound_solution
    ]
    
    results = []
    
    for algorithm in algorithms:
        start_time = time.time()
        solution = algorithm(items, weight_capacity)
        end_time = time.time()
        
        results.append({
            "algorithm": solution["algorithm"],
            "total_value": solution["total_value"],
            "total_weight": solution["total_weight"],
            "selected_items": len(solution["selected_items"]),
            "execution_time": end_time - start_time,
            "solution": solution
        })
    
    return results

def visualize_results(results):
    """Visualize the results of different algorithms."""
    # Bar chart for total value
    algorithms = [result["algorithm"] for result in results]
    values = [result["total_value"] for result in results]
    weights = [result["total_weight"] for result in results]
    times = [result["execution_time"] * 1000 for result in results]  # Convert to milliseconds
    
    # Set up figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot value comparison
    ax1.bar(algorithms, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_title('Total Value Comparison')
    ax1.set_ylabel('Total Value')
    for i, v in enumerate(values):
        ax1.text(i, v + 5, str(v), ha='center')
    
    # Plot weight comparison
    ax2.bar(algorithms, weights, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_title('Total Weight Comparison')
    ax2.set_ylabel('Total Weight (kg)')
    for i, w in enumerate(weights):
        ax2.text(i, w + 1, f"{w:.1f}", ha='center')
    
    # Plot execution time
    ax3.bar(algorithms, times, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax3.set_title('Execution Time Comparison')
    ax3.set_ylabel('Time (milliseconds)')
    ax3.set_yscale('log')  # Use log scale for better visualization
    for i, t in enumerate(times):
        ax3.text(i, t * 1.1, f"{t:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()
    
    # Create a detailed table for the report
    table_data = []
    for result in results:
        table_data.append([
            result["algorithm"],
            result["total_value"],
            f"{result['total_weight']:.1f} kg",
            result["selected_items"],
            f"{result['execution_time']*1000:.2f} ms"
        ])
    
    headers = ["Algorithm", "Total Value", "Total Weight", "Items Selected", "Execution Time"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    return table

def visualize_cargo_distribution(solution):
    """Visualize the distribution of selected cargo items."""
    selected_items = solution["selected_items"]
    
    # Group items by type
    item_types = {}
    for item in selected_items:
        item_type = item.name.split('-')[0]
        if item_type in item_types:
            item_types[item_type]["count"] += 1
            item_types[item_type]["weight"] += item.weight
            item_types[item_type]["value"] += item.value
        else:
            item_types[item_type] = {
                "count": 1,
                "weight": item.weight,
                "value": item.value
            }
    
    # Prepare data for visualization
    types = list(item_types.keys())
    counts = [item_types[t]["count"] for t in types]
    weights = [item_types[t]["weight"] for t in types]
    values = [item_types[t]["value"] for t in types]
    
    # Set up figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot count by type
    ax1.bar(types, counts, color='skyblue')
    ax1.set_title('Number of Items by Type')
    ax1.set_ylabel('Count')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot weight by type
    ax2.bar(types, weights, color='lightgreen')
    ax2.set_title('Total Weight by Type (kg)')
    ax2.set_ylabel('Weight (kg)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot value by type
    ax3.bar(types, values, color='salmon')
    ax3.set_title('Total Value by Type')
    ax3.set_ylabel('Value')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('cargo_distribution.png')
    plt.close()

def main():
    print("===== Knapsack Problem for Cargo Optimization =====")
    print("\nGenerating random cargo items...")
    
    # Generate random cargo items
    num_items = 30
    items = generate_random_cargo_items(num_items)
    
    # Display generated items
    print(f"\nGenerated {num_items} cargo items:")
    item_table = []
    for i, item in enumerate(items, 1):
        item_table.append([i, item.name, f"{item.weight} kg", f"{item.volume} m³", item.value])
    
    headers = ["ID", "Item Name", "Weight", "Volume", "Value"]
    print(tabulate(item_table, headers=headers, tablefmt="grid"))
    
    # Set weight capacity
    weight_capacity = 500  # kg
    print(f"\nTotal Weight Capacity: {weight_capacity} kg")
    
    # Compare algorithms
    print("\nComparing algorithms...")
    results = compare_algorithms(items, weight_capacity)
    
    # Visualize results
    print("\nVisualizing results...")
    result_table = visualize_results(results)
    print("\nAlgorithm Comparison Results:")
    print(result_table)
    
    # Get the best solution (assuming dynamic programming or branch and bound is best)
    best_solution = max(results, key=lambda x: x["total_value"])["solution"]
    
    # Display selected items in the best solution
    print(f"\nBest Solution ({best_solution['algorithm']}):")
    print(f"Total Value: {best_solution['total_value']}")
    print(f"Total Weight: {best_solution['total_weight']:.1f} kg out of {weight_capacity} kg")
    print(f"Weight Utilization: {(best_solution['total_weight'] / weight_capacity) * 100:.1f}%")
    print("\nSelected Items:")
    
    selected_table = []
    for i, item in enumerate(best_solution["selected_items"], 1):
        selected_table.append([i, item.name, f"{item.weight} kg", f"{item.volume} m³", item.value])
    
    print(tabulate(selected_table, headers=headers, tablefmt="grid"))
    
    # Visualize cargo distribution
    print("\nVisualizing cargo distribution...")
    visualize_cargo_distribution(best_solution)
    
    print("\nAnalysis complete. Results saved as 'algorithm_comparison.png' and 'cargo_distribution.png'")

if __name__ == "__main__":
    main()