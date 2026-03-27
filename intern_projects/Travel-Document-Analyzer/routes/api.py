from flask import Blueprint, request, jsonify, render_template_string
import os, re, io, json
from datetime import datetime
from google.cloud import vision

api_blueprint = Blueprint('api', __name__)

# === Load country requirements JSON ===
with open('travel_docs_json.json', 'r') as f:
    country_db = json.load(f)['travel_requirements']['countries']

def get_requirements_for_country(country_code):
    for country in country_db:
        if country.get('country_code', '').upper() == country_code.upper():
            return country['requirements']
    return {}

def extract_passport_fields(text):
    fields = {}

    # Passport number
    passport_match = re.search(r'\b([A-Z][0-9]{7})\b', text)
    if passport_match:
        fields['passport_number'] = passport_match.group(1)

    # Name
# Extract name from MRZ line (skip country code)
    mrz_lines = re.findall(r'[A-Z0-9<]{40,}', text)
    for line in mrz_lines:
        if line.startswith('P<'):
            line = line[2:]  # Remove 'P<'
            name_part = line[3:]  # Skip 3-letter country code
            name_clean = name_part.split('<<')[0].replace('<', ' ').strip()
            fields['name'] = re.sub(' +', ' ', name_clean)  # Clean multiple spaces
            break


    # Date of birth
    dob_match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
    if dob_match:
        fields['date_of_birth'] = dob_match.group(1)

    # Expiry date
    expiry_match = re.search(r'(Date of Expiry|Expiry Date)[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})', text, re.IGNORECASE)
    if expiry_match:
        fields['expiry_date'] = expiry_match.group(2)
    else:
        all_dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)
        if all_dates:
            try:
                fields['expiry_date'] = max(all_dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))
            except:
                pass

    # Nationality - visual zone first
    nationality_match = re.search(r'Nationality\s*[:\-]?\s*([A-Z]{3})', text, re.IGNORECASE)
    if nationality_match:
        fields['nationality'] = nationality_match.group(1).upper()
    else:
        # Fallback to MRZ line
        mrz_lines = re.findall(r'[A-Z0-9<]{40,}', text)
        for line in mrz_lines:
            match = re.search(r'[A-Z0-9]{7}<([A-Z]{3})[0-9]{6}', line)
            if match:
                fields['nationality'] = match.group(1)
                break

    return fields


def validate_passport_data(fields, country_code):
    report = {"status": "valid", "missing": [], "issues": []}

    # Look up requirements for the destination country
    country_req = get_requirements_for_country(country_code)
    passport_rules = country_req.get("passport", {})

    required_fields = passport_rules.get("required_fields", ["passport_number", "name", "date_of_birth", "expiry_date", "nationality"])
    allowed_nationalities = passport_rules.get("allowed_nationalities", [])

    for field in required_fields:
        if field not in fields or not fields[field]:
            report["missing"].append(field)

    try:
        expiry = datetime.strptime(fields.get("expiry_date", ""), "%d/%m/%Y")
        if expiry < datetime.now():
            report["issues"].append("Passport is expired")
    except:
        report["issues"].append("Invalid expiry date format")

    if allowed_nationalities and fields.get("nationality") not in allowed_nationalities:
        report["issues"].append("Nationality not allowed")

    if report["missing"] or report["issues"]:
        report["status"] = "invalid"

    return report


# === VISA ===
def extract_visa_fields(text):
    fields = {}
    lines = text.splitlines()

    print("=== OCR Lines ===")
    for i, line in enumerate(lines):
        print(f"{i}: {line}")
    print("=================")


    # === Name extraction ===
    surname = None
    given = None
    for i, line in enumerate(lines):
        if "Surname" in line:
            for j in range(1, 4):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if candidate.isalpha() and len(candidate) < 30:
                        surname = candidate
                        break
        if "Given Name" in line:
            for j in range(1, 4):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if candidate.isalpha() and len(candidate) < 30:
                        given = candidate
                        break

    if surname and given:
        fields['name'] = f"{surname} {given}"
    else:
        mrz = re.search(r'VN[A-Z]{3}<([A-Z<]+)', text)
        if mrz:
            fields['name'] = mrz.group(1).replace('<', ' ').strip()

    # === Expiry date extraction ===
    for i, line in enumerate(lines):
        if "Expiration" in line or "Expiry" in line:
            for j in range(1, 3):
                if i + j < len(lines):
                    match = re.search(r'\b\d{2}[A-Z]{3}\d{4}\b', lines[i + j])
                    if match:
                        try:
                            dt = datetime.strptime(match.group(), "%d%b%Y")
                            fields['expiry_date'] = dt.strftime("%d/%m/%Y")
                            break
                        except:
                            pass
            break

    # === Nationality extraction ===
    # === Nationality Extraction (visual zone or next-line) ===
    nationality_found = False
    for i, line in enumerate(lines):
        if "Nationality" in line:
            # Check same line
            match = re.search(r'Nationality\s*[:\-]?\s*([A-Z]{3})', line, re.IGNORECASE)
            if match:
                fields['nationality'] = match.group(1).upper()
                nationality_found = True
                break
            # Look next 2 lines
            for j in range(1, 3):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    # Check if line ends with 3-letter country code
                    match = re.search(r'\b([A-Z]{3})\b$', next_line)
                    if match:
                        fields['nationality'] = match.group(1).upper()
                        nationality_found = True
                        break
            if nationality_found:
                break


    # === Issuing country detection ===
    for country in ["UNITED STATES", "UNITED STATES OF AMERICA", "USA", "AMERICA"]:
        if country in text.upper():
            fields['issuing_country'] = "UNITED STATES"
            break

    # === Visa number detection ===
    candidates = re.findall(r'\b[A-Z][0-9]{6,10}\b', text)
    ignored_prefixes = ['A']  # ignore common control number prefixes
    preferred_length_range = range(7, 10)

    visa_number = None
    for candidate in candidates:
        if (
            not any(candidate.startswith(pref) for pref in ignored_prefixes)
            and len(candidate) in preferred_length_range
        ):
            visa_number = candidate
            break

    if not visa_number:
        # Fallback to control number
        control = re.search(r'Control Number\s*[:\-]?\s*([0-9]{10,})', text, re.IGNORECASE)
        if control:
            visa_number = control.group(1)

    if visa_number:
        fields['visa_number'] = visa_number

    return fields



def validate_visa_data(fields, country_code):
    report = {"status": "valid", "missing": [], "issues": []}

    country_req = get_requirements_for_country(country_code)
    visa_rules = country_req.get("visa", {})

    required_fields = visa_rules.get("required_fields", ["visa_number", "name", "expiry_date", "nationality", "issuing_country"])
    allowed_nationalities = visa_rules.get("allowed_nationalities", [])

    for field in required_fields:
        if not fields.get(field):
            report["missing"].append(field)

    try:
        expiry = datetime.strptime(fields.get("expiry_date", ""), "%d/%m/%Y")
        if expiry < datetime.now():
            report["issues"].append("Visa is expired")
    except:
        report["issues"].append("Invalid expiry date format")

    if allowed_nationalities and fields.get("nationality") not in allowed_nationalities:
        report["issues"].append("Nationality not allowed")

    if report["missing"] or report["issues"]:
        report["status"] = "invalid"

    return report


# === ID CARD ===
def extract_id_card_fields(text):
    fields = {}
    name_match = re.search(r'Name\s*[:\-]?\s*([A-Z ]{3,})', text)
    if name_match:
        fields['name'] = name_match.group(1).strip()

    dob_match = re.search(r'\b(?:DOB|Date of Birth)[:\-]?\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    if dob_match:
        fields['date_of_birth'] = dob_match.group(1)

    id_match = re.search(r'\b(ID Number|ID No|ID#)[:\-]?\s*([A-Z0-9]{6,})', text, re.IGNORECASE)
    if id_match:
        fields['id_number'] = id_match.group(2)

    nat = re.search(r'Nationality\s*[:\-]?\s*([A-Z]{3,})', text, re.IGNORECASE)
    if nat:
        fields['nationality'] = nat.group(1).upper()

    return fields


def validate_id_card_data(fields):
    required = ['name', 'date_of_birth', 'id_number']
    report = {"status": "valid", "missing": []}

    for field in required:
        if not fields.get(field):
            report["missing"].append(field)

    if report["missing"]:
        report["status"] = "incomplete"

    return report

# === BOARDING PASS ===
def extract_boarding_pass_fields(text):
    fields = {}
    lines = text.splitlines()

    # Name
    for i, line in enumerate(lines):
        if "Passenger" in line:
            for j in range(1, 3):
                if i + j < len(lines):
                    name = lines[i + j].strip()
                    if len(name.split()) >= 2:
                        fields["name"] = name
                        break

    # From and To (multi-line tolerant)
    for i, line in enumerate(lines):
        if "From" in line:
            from_candidate = line.split("From")[-1].strip(":").strip()
            if not from_candidate and i + 1 < len(lines):
                from_candidate = lines[i + 1].strip()
            if from_candidate:
                fields["from"] = from_candidate
        if "To" in line:
            to_candidate = line.split("To")[-1].strip(":").strip()
            if not to_candidate and i + 1 < len(lines):
                to_candidate = lines[i + 1].strip()
            if to_candidate:
                fields["to"] = to_candidate

    # Flight number
    flight_match = re.search(r'\b[A-Z]{2}\d{3,4}\b', text)
    if flight_match:
        fields["flight_number"] = flight_match.group(0)

    # Seat number
    seat_match = re.search(r'\bSeat[:\s]*([0-9]{1,2}[A-Z])', text, re.IGNORECASE)
    if seat_match:
        fields["seat"] = seat_match.group(1)

    # Date (format: 06 DEC 20)
    date_match = re.search(r'\b(\d{2}\s+[A-Z]{3}\s+\d{2})\b', text)
    if date_match:
        try:
            parsed_date = datetime.strptime(date_match.group(1), "%d %b %y")
            fields["date"] = parsed_date.strftime("%d/%m/%Y")
        except:
            fields["date"] = date_match.group(1)

    return fields




def validate_boarding_pass_data(fields):
    required = ['name', 'flight_number', 'from', 'to', 'seat', 'date']
    report = {"status": "valid", "missing": []}

    for field in required:
        if not fields.get(field):
            report["missing"].append(field)

    if report["missing"]:
        report["status"] = "incomplete"

    return report

def extract_pan_fields(text):
    fields = {}
    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', text)
    if pan_match:
        fields['pan_number'] = pan_match.group(1)

    name_match = re.search(r'Name\s*[:\-]?\s*([A-Z ]+)', text, re.IGNORECASE)
    if name_match:
        fields['name'] = name_match.group(1).strip()

    dob_match = re.search(r'Date of Birth\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', text)
    if dob_match:
        fields['date_of_birth'] = dob_match.group(1)

    return fields
def validate_pan_data(fields):
    required = ["pan_number", "name", "date_of_birth"]
    report = {"status": "valid", "missing": []}

    for field in required:
        if not fields.get(field):
            report["missing"].append(field)

    if report["missing"]:
        report["status"] = "incomplete"

    return report
def extract_aadhaar_fields(text):
    fields = {}

    # Aadhaar number (12-digit)
    aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
    if aadhaar_match:
        fields['aadhaar_number'] = aadhaar_match.group().replace(' ', '')

    # Name (appears before DOB)
    name_match = re.search(r'DOB\s*[:\-]?\s*\d{2}/\d{2}/\d{4}\s*\n?([A-Z][A-Z\s]+)', text, re.IGNORECASE)
    if name_match:
        fields['name'] = name_match.group(1).strip().title()
    else:
        # Fallback
        name_line = re.search(r'Darakshan.*', text)
        if name_line:
            fields['name'] = name_line.group().strip()

    # Date of Birth
    dob_match = re.search(r'DOB\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', text)
    if dob_match:
        fields['date_of_birth'] = dob_match.group(1)

    # Gender
    gender_match = re.search(r'\b(MALE|FEMALE|OTHER)\b', text, re.IGNORECASE)
    if gender_match:
        fields['gender'] = gender_match.group(1).capitalize()

    return fields


def validate_aadhaar_data(fields):
    required = ['aadhaar_number', 'name', 'date_of_birth']
    report = {"status": "valid", "missing": []}

    for field in required:
        if not fields.get(field):
            report["missing"].append(field)

    if report["missing"]:
        report["status"] = "incomplete"

    return report

# === Extract/validate functions (same as before, shorten for brevity) ===
# paste your existing `extract_*_fields()` and `validate_*_data()` functions here...

# You already pasted them correctly earlier, just keep them unchanged.

# === Extra document checker ===
def check_additional_requirements(documents_uploaded, target_country_code):
    required_docs = get_requirements_for_country(target_country_code)

    needed = []
    # Check if passport is required but not uploaded
    if required_docs.get("passport", {}).get("required", False) and 'passport' not in documents_uploaded:
        needed.append("passport")

    if required_docs.get("visa", {}).get("required", False) and 'visa' not in documents_uploaded:
        needed.append("visa")

    health_docs = required_docs.get("health_documents", {})
    for key, val in health_docs.items():
        if str(val.get("required", "")).lower() in ["mandatory", "required"] and key not in documents_uploaded:
            needed.append(key)

    return needed

# === Upload route ===
@api_blueprint.route('/upload', methods=['POST'])
def upload_documents():
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    # ✅ Get destination country code from form-data or default to "IND"
    country_code = request.form.get('country', 'IND').upper()

    files = request.files.getlist('document')
    results = []

    client = vision.ImageAnnotatorClient()
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)

    for file in files:
        if file.filename == '':
            continue

        try:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            save_path = os.path.join(upload_folder, filename)
            file.save(save_path)

            with io.open(save_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            extracted_text = response.full_text_annotation.text

            text_lower = extracted_text.lower()
            fname_lower = file.filename.lower()
            print("==== OCR TEXT START ====")
            print(extracted_text)
            print("==== OCR TEXT END ====")


            # === Document Type Detection ===
            if "aadhaar" in text_lower or "uidai" in text_lower or re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', extracted_text):
                doc_type = "aadhaar_card"
                fields = extract_aadhaar_fields(extracted_text)
                validation = validate_aadhaar_data(fields)

            elif "visa" in text_lower or "control number" in text_lower:
                doc_type = "visa"
                fields = extract_visa_fields(extracted_text)
                validation = validate_visa_data(fields, country_code)

            elif "boarding pass" in text_lower or "gate" in text_lower:
                doc_type = "boarding_pass"
                fields = extract_boarding_pass_fields(extracted_text)
                validation = validate_boarding_pass_data(fields)

            elif "income tax" in text_lower or "permanent account number" in text_lower or "pan" in text_lower:
                doc_type = "pan_card"
                fields = extract_pan_fields(extracted_text)
                validation = validate_pan_data(fields)

            elif "id" in text_lower or "identity" in text_lower:
                doc_type = "id_card"
                fields = extract_id_card_fields(extracted_text)
                validation = validate_id_card_data(fields)

            else:
                doc_type = "passport"
                fields = extract_passport_fields(extracted_text)
                validation = validate_passport_data(fields, country_code)

            results.append({
                'filename': file.filename,
                'document_type': doc_type,
                'extracted_fields': fields,
                'validation': validation
            })

        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results})



# === Routes ===
@api_blueprint.route('/upload-form', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        if 'document' not in request.files:
            return "No file uploaded"

        files = request.files.getlist('document')
        country_code = request.form.get("country_code", "IND")
        results = []

        client = vision.ImageAnnotatorClient()
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)

        for file in files:
            if file.filename == '':
                continue

            try:
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                save_path = os.path.join(upload_folder, filename)
                file.save(save_path)

                with io.open(save_path, 'rb') as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
                response = client.document_text_detection(image=image)
                extracted_text = response.full_text_annotation.text
                text_lower = extracted_text.lower()

                # Detect document type
                if "visa" in text_lower or "control number" in text_lower:
                    doc_type = "visa"
                    fields = extract_visa_fields(extracted_text)
                    validation = validate_visa_data(fields, country_code)
                elif "boarding pass" in text_lower or "gate" in text_lower:
                    doc_type = "boarding_pass"
                    fields = extract_boarding_pass_fields(extracted_text)
                    validation = validate_boarding_pass_data(fields)
                elif "income tax" in text_lower or "pan" in text_lower:
                    doc_type = "pan_card"
                    fields = extract_pan_fields(extracted_text)
                    validation = validate_pan_data(fields)
                elif "id" in file.filename.lower() or "identity" in text_lower:
                    doc_type = "id_card"
                    fields = extract_id_card_fields(extracted_text)
                    validation = validate_id_card_data(fields)
                else:
                    doc_type = "passport"
                    fields = extract_passport_fields(extracted_text)
                    validation = validate_passport_data(fields, country_code)

                results.append({
                    'filename': file.filename,
                    'document_type': doc_type,
                    'extracted_fields': fields,
                    'validation': validation
                })

            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })

        # HTML template
        template = """
        <html>
        <head>
            <title>Document Upload</title>
            <style>
                body { font-family: Arial; padding: 20px; background: #f9f9f9; }
                h2 { color: #333; }
                .card {
                    background: white; 
                    border-radius: 6px; 
                    padding: 15px; 
                    margin: 15px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .valid { color: green; font-weight: bold; }
                .invalid { color: red; font-weight: bold; }
                pre { background: #eee; padding: 10px; }
            </style>
        </head>
        <body>
            <h2>Upload Documents</h2>
            <form method="POST" enctype="multipart/form-data">
                <label>Destination Country Code:</label>
                <input type="text" name="country_code" value="IND" required><br><br>
                <input type="file" name="document" multiple required>
                <button type="submit">Upload</button>
            </form>

            {% if results %}
                <h3>Results</h3>
                {% for res in results %}
                    <div class="card">
                        <strong>File:</strong> {{ res.filename }}<br>
                        <strong>Type:</strong> {{ res.document_type|capitalize }}<br><br>
                        {% if res.error %}
                            <span class="invalid">Error: {{ res.error }}</span>
                        {% else %}
                            <strong>Extracted Fields:</strong>
                            <pre>{{ res.extracted_fields | tojson(indent=2) }}</pre>

                            <strong>Validation:</strong>
                            <p>Status: 
                                {% if res.validation.status == 'valid' %}
                                    <span class="valid">VALID ✅</span>
                                {% else %}
                                    <span class="invalid">INVALID ❌</span>
                                {% endif %}
                            </p>
                            {% if res.validation.missing %}
                                <strong>Missing Fields:</strong> {{ res.validation.missing }}<br>
                            {% endif %}
                            {% if res.validation.issues %}
                                <strong>Issues:</strong> {{ res.validation.issues }}<br>
                            {% endif %}
                        {% endif %}
                    </div>
                {% endfor %}
            {% endif %}
        </body>
        </html>
        """
        return render_template_string(template, results=results)

    return '''
        <h2>Upload Documents</h2>
        <form method="POST" enctype="multipart/form-data">
            <label>Destination Country Code:</label>
            <input type="text" name="country_code" value="IND" required><br><br>
            <input type="file" name="document" accept="image/*" multiple required>
            <button type="submit">Upload</button>
        </form>
    '''


@api_blueprint.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'API is alive'})


