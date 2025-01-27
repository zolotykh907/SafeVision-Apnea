import xml.etree.ElementTree as ET

def get_attrubuts(file_path, space = 'ScoringData', 
                  namespace = 'http://www.respironics.com/PatientStudy.xsd', 
                  tag="Events", family='Respiratory', tip='ObstructiveApnea'):
    '''
    root - корень файла xml
    space - где надо искать аттрибуты
    namespace - пространство имен, где все искать
    tag - на что должен заканчиваться тег в пространстве имен
    '''
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = root.find(f'ns:{space}', {'ns': namespace})

    if data is None:
        return []
    
    target_arr = []

    events = None
    for child in data:
        if child.tag.endswith(tag):
            events = child
            break

    if events is None:
        return []

    for event in events:
        event_attributs = event.attrib
        if event_attributs['Family'] == family and event_attributs['Type'] == tip:
            target_arr.append(event_attributs)

    return target_arr


apnea_attributs = get_attrubuts('/var/data/apnea/rml/00000993-100507.rml')