terra_incognita_classes = {
    0: "Bird",
    1: "Bobcat",
    2: "Cat",
    3: "Coyote",
    4: "Dog",
    5: "Opossum",
    6: "Rabbit",
    7: "Raccoon",
    8: "Rodent",
    9: "Empty",  # No animal in the image
}

pacs_classes = {
    0: "Dog",
    1: "Elephant",
    2: "Giraffe",
    3: "Guitar",
    4: "Horse",
    5: "House",
    6: "Person"
}

# Common VLCS splits use these 5 classes, though your setup may vary.
vlcs_classes = {
    0: "Bird",
    1: "Car",
    2: "Chair",
    3: "Dog",
    4: "Person"
}

# WILDS Camelyon is typically a binary tumor classification: normal vs. tumor.
wilds_camelyon_classes = {
    0: "normal",
    1: "tumor"
}

# WILDS FMoW commonly has 62 classes (1 for each facility/location type).
# Below is one commonly used mapping. You may need to adjust indices or names
# if your split or label definitions differ.
wilds_fmow_classes = {
    0:  "Airport",
    1:  "Airport Hangar",
    2:  "Airport Terminal",
    3:  "Amusement Park",
    4:  "Aquaculture",
    5:  "Archaeological Site",
    6:  "Barn",
    7:  "Border Checkpoint",
    8:  "Burial Site",
    9:  "Car Dealership",
    10: "Construction Site",
    11: "Crop Field",
    12: "Dam",
    13: "Debris or Rubble",
    14: "Educational Institution",
    15: "Electric Substation",
    16: "Factory or Powerplant",
    17: "Fire Station",
    18: "Fishing Facility",
    19: "Flooded Road",
    20: "Fountain",
    21: "Gas Station",
    22: "Golf Course",
    23: "Ground Transportation Station",
    24: "Helipad",
    25: "Hospital",
    26: "Impoverished Settlement",
    27: "Interchange",
    28: "Lake or Pond",
    29: "Lighthouse",
    30: "Military Facility",
    31: "Multi-Unit Residential",
    32: "Nuclear Reactor",
    33: "Office Building",
    34: "Oil or Gas Facility",
    35: "Park",
    36: "Parking Lot or Garage",
    37: "Place of Worship",
    38: "Police Station",
    39: "Port",
    40: "Prison",
    41: "Race Track",
    42: "Railway Bridge",
    43: "Recreational Facility",
    44: "Road Bridge",
    45: "Runway",
    46: "Shipyard",
    47: "Shopping Mall",
    48: "Smokestack",
    49: "Solar Farm",
    50: "Space Facility",
    51: "Stadium",
    52: "Storage Tank",
    53: "Surface Mine",
    54: "Swimming Pool",
    55: "Toll Booth",
    56: "Tower",
    57: "Tunnel Opening",
    58: "Waste Disposal",
    59: "Water Treatment Facility",
    60: "Wind Farm",
    61: "Zoo"
}

cxr_no_finding_classes = {
    0: "No Findings",
    1: "Finding"
}

def get_class_names_dict(dataset_name):
    if dataset_name == "TerraIncognita":
        return terra_incognita_classes
    elif dataset_name == "PACS":
        return pacs_classes
    elif dataset_name == "VLCS":
        return vlcs_classes
    elif dataset_name == "WILDSCamelyon":
        return wilds_camelyon_classes
    elif dataset_name == "WILDSFMoW":
        return wilds_fmow_classes
    elif dataset_name == "CXR_No_Finding":
        return cxr_no_finding_classes
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
