import numpy as np
from enum import Enum, IntEnum


class AttributeTypes(IntEnum):
    UNORDERED: 0
    ORDERED: 1
    CONTINUOUS: 2


class Attribute():
    def __init__(self, att_index, name, description):
        self.att_index = att_index
        self.name = name
        self.description = description
        self.values_dict = {}
        self.attribute_type = AttributeTypes.CONTINUOUS

    def set_value_dict(self, d):
        self.values_dict = d
        self.number_of_values = len(d)

    def get_value_description(self, value):
        return self.values_dict[value]

    def get_key_index(self, key):
        return list(self.values_dict.keys()).index(key)

    def one_hot_encode(self, value):
        classes = self.number_of_values
        index = list(self.values_dict.keys()).index(value)
        return np.eye(classes)[index]


class DataLookup():
    def __init__(self, fileName="train.csv"):
        attributes = []
        ID = Attribute(
            0, "Id", "Index of the Record")
        MSSubClass = Attribute(
            1, "MSSubClass", "Identifies the type of dwelling involved in the sale.")
        MSSubClass.set_value_dict({
            20:	"1-STORY 1946 & NEWER ALL STYLES",
            30:	"1-STORY 1945 & OLDER",
            40:	"1-STORY W/FINISHED ATTIC ALL AGES",
            45:	"1-1/2 STORY - UNFINISHED ALL AGES",
            50:	"1-1/2 STORY FINISHED ALL AGES",
            60:	"2-STORY 1946 & NEWER",
            70:	"2-STORY 1945 & OLDER",
            75:	"2-1/2 STORY ALL AGES",
            80:	"SPLIT OR MULTI-LEVEL",
            85:	"SPLIT FOYER",
            90:	"DUPLEX - ALL STYLES AND AGES",
            120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
            150: "1-1/2 STORY PUD - ALL AGES",
            160: "2-STORY PUD - 1946 & NEWER",
            180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
            190: "2 FAMILY CONVERSION - ALL STYLES AND AGES"
        })
        MSSubClass.attribute_type = AttributeTypes.UNORDERED
        attributes.append(MSSubClass)

        MSZoning = Attribute(
            2, "MSZoning", "Identifies the general zoning classification of the sale.")
        MSZoning.set_value_dict({
            "A": "Agriculture",
            "C": "Commercial",
            "FV": "Floating Village Residential",
            "I": "Industrial",
            "RH": "Residential High Density",
            "RL": "Residential Low Density",
            "RP": "Residential Low Density Park",
            "RM": "Residential Medium Density",
        })
        MSZoning.attribute_type = AttributeTypes.UNORDERED
        attributes.append(MSZoning)

        LotFrontage = Attribute(
            3, "LotFrontage", "Linear feet of street connected to property.")
        attributes.append(LotFrontage)
        
        LotArea = Attribute(4,"LotArea","Lot size in square feet")
        attributes.append(LotArea)
        
        Street = Attribute(5, "Street", "Type of road access to property")
        Street.set_value_dict({
            "Grvl": "Gravel",
            "Pave": "Paved"
        })
        Street.attribute_type = AttributeTypes.UNORDERED
        attributes.append(Street)
        
        Alley = Attribute(6, "Alley", "Type of alley access to property")
        Alley.set_value_dict({
            "Grvl": "Gravel",
            "Pave": "Paved",
            "NA": "No alley access"
        })
        Alley.attribute_type = AttributeTypes.UNORDERED
        attributes.append(Alley)
        
        LotShape = Attribute(7, "LotShape", "General shape of property")
        LotShape.set_value_dict({
            "Reg": "Regular",
            "IR1": "Slightly irregular",
            "IR2": "Moderately Irregular",
            "IR3": "Irregular"
        })
        LotShape.attribute_type = AttributeTypes.UNORDERED
        attributes.append(LotShape)
        
        LandContour = Attribute(8, "LandContour", "Flatness of the property")
        LandContour.set_value_dict({
            "Lvl": "Near Flat/Level",
            "Bnk": "Banked - Quick and significant rise from street grade to building",
            "HLS": "Hillside - Significant slope from side to side",
            "Low": "Depression"
        })
        LandContour.attribute_type = AttributeTypes.UNORDERED
        attributes.append(LandContour)
        
        Utilities = Attribute(9, "Utilities", "Type of utilities available")
        Utilities.set_value_dict({
            "Lvl": "Near Flat/Level",
            "Bnk": "Banked - Quick and significant rise from street grade to building",
            "HLS": "Hillside - Significant slope from side to side",
            "Low": "Depression"
        })
        Utilities.attribute_type = AttributeTypes.UNORDERED
        attributes.append(Utilities)
        
        LotConfig = Attribute(10, "LotConfig", "Lot configuration")
        LotConfig.set_value_dict({
            "Inside": "Inside lot",
            "Corner": "Corner lot",
            "CulDSac": "Cul-de-sac",
            "FR2": "Frontage on 2 sides of property",
            "FR3": "Frontage on 3 sides of property"
        }) 
        LotConfig.attribute_type = AttributeTypes.UNORDERED
        attributes.append(LotConfig)
        
        LandSlope = Attribute(10, "LandSlope", "Slope of property")
        LandSlope.set_value_dict({
            "Gtl": "Gentle slope",
            "Mod": "Moderate Slope",
            "Sev": "Severe Slope"
        })
        LandSlope.attribute_type = AttributeTypes.UNORDERED
        attributes.append(LandSlope)
        
        self.attributes = attributes
        

    def initHeaders(self, headers):
        self.headers = headers

    def get_header(self, idx):
        return self.headers[idx]

    def convert_data(self, column, value):
        att = self.attributes[column]        
        return att.one_hot_encode(value)


def main():
    d = DataLookup()


if __name__ == "__main__":
    main()
