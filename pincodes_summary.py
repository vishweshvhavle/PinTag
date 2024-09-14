import csv
from collections import defaultdict
from tqdm import tqdm

# Initialize dictionaries for storing sets and counts
state_data = defaultdict(lambda: [set(), set(), set()])
count = 0
unique_pincodes = set()  # Set to track unique pincodes

# Read the CSV file and process each row
with open('pincodes.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in tqdm(reader, desc="Processing Pincodes"):
        count += 1
        state = row['State']
        pincode = row['Pincode']

        # Add pincode to unique pincodes set
        unique_pincodes.add(pincode)

        # Split pincode into three parts
        part1, part2, part3 = pincode[:2], pincode[2:4], pincode[4:6]

        # Add values to the corresponding sets for the state
        state_data[state][0].add(part1)
        state_data[state][1].add(part2)
        state_data[state][2].add(part3)

# Generate a new CSV file with unique sets and counts for each state
with open('pincode_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['State', 'Set 1', 'Count 1', 'Set 2', 'Count 2', 'Set 3', 'Count 3'])

    for state, sets in state_data.items():
        writer.writerow([state,
                         ','.join(sets[0]), len(sets[0]),
                         ','.join(sets[1]), len(sets[1]),
                         ','.join(sets[2]), len(sets[2])])

# Calculate the union of all sets for part 1, part 2, and part 3
set1_union = set()
set2_union = set()
set3_union = set()

for sets in state_data.values():
    set1_union.update(sets[0])
    set2_union.update(sets[1])
    set3_union.update(sets[2])

# Print the union of all sets and their total counts
print(f"Union of Set 1: {set1_union}, Total Count: {len(set1_union)}")
print(f"Union of Set 2: {set2_union}, Total Count: {len(set2_union)}")
print(f"Union of Set 3: {set3_union}, Total Count: {len(set3_union)}")

# Print the total number of unique pincodes and total processed pincodes
print(f"Total Pincodes Processed: {count}")
print(f"Total Unique Pincodes: {len(unique_pincodes)}")
