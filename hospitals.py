import pandas as pd
import json
from os.path import join, dirname
import re


def get_suitable_hospitals(specialty, region):          
    """
    This method gets suitable hospitals for a given 
    specialty and region (İstanbul Avrupa - Anadolu)
    
    Parameters
    ----------
    specialty : string
        Medical Specialty as string.
    region : string
        Region as string
        
    Returns
    -------
    list
        Hospital name-coordinate tuple list.
    list
        Names of doctors that should be displayed.
    """
    with open('data\\healthcare_json.json', encoding='utf8') as f:
        data = json.load(f)
    
    turkishLabels = ['Deri ve Zührevi Hastalıkları (Cildiye)', 'İç Hastalıkları (Dahiliye)', 'Nöroloji','Kadın Hastalıkları ve Doğum', 'Göz Hastalıkları', 'Ortopedi ve Travmatoloji','Kulak Burun Boğaz Hastalıkları', 'Çocuk Sağlığı ve Hastalıkları', 'Ruh Sağlığı ve Hastalıkları','Radyoloji', 'Genel Cerrahi', 'Üroloji']
    englishLabels = ['Dermatology', 'Internal Medicine', 'Neurology','Obstetrics & Gynecology', 'Ophthalmology', 'Orthopaedic Surgery','Otolaryngology', 'Pediatrics', 'Psychiatry','Radiology-Diagnostic', 'Surgery-General', 'Urology']
    turkishLabels = [item.lower() for item in turkishLabels]
    englishLabels = [item.lower() for item in englishLabels]
    
    region = "(" + region + ")"
    
    specialty = specialty.lower()
    specialty = turkishLabels[englishLabels.index(specialty)]
    
    hospital_list = []
    for item in data['Hospital']:
        if(specialty == item['hospitalName'].lower() and region.lower() in item['clinicName'].lower()):
            hospital_list.append(item['clinicName'])
    
    hospital_list = set(hospital_list)
    
    hospital_list_with_coords = []
    for item in hospital_list:
        for i in range(len(data['Location'])):
            if item == data['Location'][i]['hospitalTitle']:
                hospital_list_with_coords.append(data['Location'][i])

#     hospital_list_with_coords = set(list(hospital_list_with_coords))
#     print(hospital_list_with_coords)
    hospital_list_with_coords = sorted(hospital_list_with_coords, key=lambda x : x['hospitalTitle'])
    hospital_list_with_coords = pd.DataFrame(hospital_list_with_coords).values.tolist()

    displayDoctors = []

    for i in range(len(hospital_list_with_coords)):
        hospital_list_with_coords[i].append(False)
        hospital_list_with_coords[i].append(['doc1','doc2','doc3'])
        displayDoctors.append(False)
    # 
    
    # Assign doctor for chosen specialty
    for i in range(len(hospital_list_with_coords)):
        doctor_list = []
        for j in range(len(data['Hospital'])):
            if(hospital_list_with_coords[i][0] == data['Hospital'][j]['clinicName'] and data['Hospital'][j]['hospitalName'].lower() == specialty.lower()):
                doctor_list.append(data['Hospital'][j]['doctorName'])

        hospital_list_with_coords[i][4] = sorted(list(set(doctor_list)), key=lambda x : x)
    
    # Remove duplicate hospitals
    for count,item in enumerate(hospital_list_with_coords):
        current_hospital = item[0]
        for j in range(len(hospital_list_with_coords) - count):
            if(j != 0 and hospital_list_with_coords[-j][0] == current_hospital):
                hospital_list_with_coords.pop(-j)
    
    return hospital_list_with_coords, displayDoctors