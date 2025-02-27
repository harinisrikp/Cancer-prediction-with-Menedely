import pandas as pd
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Medical terms dictionary remains the same
MEDICAL_TERMS = {
    'Adenocarcinoma': {
        'simple': "a type of cancer",
        'explanation': "This is a type of cancer that starts in the cells that produce mucus (the slippery substance that protects your organs). Think of it like bad cells growing in the layer that normally makes protective slime in your body."
    },
    'Squamous Cell Carcinoma': {
        'simple': "a type of skin cancer",
        'explanation': "This is a cancer that starts in the flat cells that cover the surface of your cervix. Imagine your skin has layers like an onion - this cancer starts in the outer layers."
    },
    'Biopsy': {
        'simple': "tissue sample test",
        'explanation': "This is when doctors take a tiny piece of tissue (like taking a very small sample) to look at it under a microscope to check if there are any cancer cells."
    },
    'Stage': {
        'simple': "how far the cancer has spread",
        'explanation': "Cancer stages tell us how much the cancer has grown and if it has spread. Think of it like levels in a video game - higher stages mean the cancer has progressed further."
    },
    'HIV': {
        'simple': "immune system virus",
        'explanation': "HIV is a virus that affects your immune system (your body's defense system against illness). It's important to know about HIV because it can affect how well your body fights other diseases."
    },
    'CXR': {
        'simple': "chest X-ray",
        'explanation': "A chest X-ray is like taking a picture of the inside of your chest using special rays that can see through skin. It's like having a super-power camera that can look inside your body!"
    },
    'US': {
        'simple': "ultrasound",
        'explanation': "An ultrasound uses sound waves to create pictures of the inside of your body. It's the same technology used to see babies before they're born!"
    },
    'CT': {
        'simple': "detailed body scan",
        'explanation': "A CT scan is like taking many X-rays from different angles and putting them together to make a 3D picture of your body. Imagine taking photos of a house from every side and then making a 3D model from those pictures."
    },
    'MRI': {
        'simple': "magnetic body scan",
        'explanation': "An MRI uses powerful magnets and radio waves to take detailed pictures of the inside of your body. It's like having a super-detailed camera that can show exactly what's going on inside."
    }
}

# Enhanced treatment descriptions with specific use cases
TREATMENT_DESCRIPTIONS = {
    'Chemotherapy': {
        'name': "Chemotherapy",
        'explanation': "This is like sending special medicine through your blood to fight cancer cells. Think of it as tiny soldiers traveling through your body to find and stop cancer cells from growing.",
        'best_for': {
            'early_stage': "In early stages, chemotherapy can help destroy cancer cells before they spread.",
            'advanced_stage': "For advanced stages, chemotherapy can help shrink tumors and control cancer growth throughout the body.",
            'with_hiv': "The dosage and timing might be adjusted if you're HIV positive to protect your immune system."
        }
    },
    'Radiation Therapy': {
        'name': "Radiation Therapy",
        'explanation': "This uses special energy beams (like super-focused light) to target and destroy cancer cells. It's like using a very precise laser to remove the harmful cells.",
        'best_for': {
            'early_stage': "Radiation works great for targeting specific areas in early stages.",
            'advanced_stage': "Can be used to shrink tumors or manage symptoms in advanced stages.",
            'with_hiv': "Often well-tolerated even with HIV, but requires careful monitoring."
        }
    },
    'Immunotherapy': {
        'name': "Immunotherapy",
        'explanation': "This treatment helps your body's own defense system (immune system) become stronger to fight cancer cells. It's like training your body's natural warriors to better fight the cancer.",
        'best_for': {
            'early_stage': "Can help prevent cancer from coming back after other treatments.",
            'advanced_stage': "Might help when other treatments haven't worked.",
            'with_hiv': "Requires very careful monitoring with HIV due to immune system involvement."
        }
    },
    'Hormone Therapy': {
        'name': "Hormone Therapy",
        'explanation': "This treatment changes how hormones (your body's chemical messengers) work to stop cancer from growing. It's like changing the messages your body sends to prevent cancer cells from getting stronger.",
        'best_for': {
            'early_stage': "Can help prevent cancer growth in hormone-sensitive cancers.",
            'advanced_stage': "May help control cancer spread even in later stages.",
            'with_hiv': "Generally safe with HIV but requires regular monitoring."
        }
    },
    'Targeted Therapy': {
        'name': "Targeted Therapy",
        'explanation': "This is a super-specific treatment that attacks particular parts of cancer cells. Imagine having a key that fits only one lock - this treatment only affects the cancer cells while leaving healthy cells alone.",
        'best_for': {
            'early_stage': "Can be very effective when specific cancer markers are present.",
            'advanced_stage': "May work well even in advanced stages if the cancer has the right targets.",
            'with_hiv': "Often a good choice with HIV as it's less likely to affect the immune system."
        }
    }
}

# Stage information dictionary
STAGE_INFO = {
    'Stage 2': {
        'description': "The cancer is still relatively early and contained in one area.",
        'prognosis': "There's a good chance of successful treatment at this stage.",
        'typical_findings': "Usually shows up clearly on imaging tests but hasn't spread far."
    },
    'Stage 3': {
        'description': "The cancer has grown larger but is still mainly in one area.",
        'prognosis': "Treatment can still be very effective at this stage.",
        'typical_findings': "Multiple imaging tests help show the extent of the cancer."
    },
    'Stage 4': {
        'description': "The cancer has started to spread to nearby areas.",
        'prognosis': "Treatment focuses on controlling the cancer and managing symptoms.",
        'typical_findings': "Several imaging tests are usually needed to track spread."
    },
    'Stage 6': {
        'description': "The cancer has spread more extensively.",
        'prognosis': "Treatment focuses on controlling symptoms and improving quality of life.",
        'typical_findings': "Multiple imaging tests show cancer in different areas."
    },
    'Stage 7': {
        'description': "The cancer has spread to multiple areas in the body.",
        'prognosis': "Treatment focuses on comfort and maintaining quality of life.",
        'typical_findings': "Extensive imaging shows cancer in multiple locations."
    },
    'Stage 8': {
        'description': "The cancer is causing serious complications.",
        'prognosis': "Treatment focuses on managing symptoms and complications.",
        'typical_findings': "Imaging shows significant impact on affected organs."
    },
    'Stage 9': {
        'description': "The cancer is very advanced.",
        'prognosis': "Treatment focuses on comfort care and symptom management.",
        'typical_findings': "Imaging confirms extensive spread and complications."
    }
}

def get_stage_explanation(stage, imaging_results):
    """Generate personalized staging explanation based on tests and findings"""
    stage_number = stage.split(':')[0].strip()
    if stage_number not in STAGE_INFO:
        return "The staging information is not available."
    
    info = STAGE_INFO[stage_number]
    explanation = [
        f"\n🔍 Understanding the Cancer Stage:",
        f"• {info['description']}",
        f"• What this means: {info['prognosis']}",
        "\nHow we determined this stage:"
    ]
    
    # Add information about what the imaging tests showed
    tests_done = []
    if imaging_results['cxrdone'] == 'CXR Done':
        tests_done.append("chest X-ray")
    if imaging_results['ctdone'] == 'CT Done':
        tests_done.append("CT scan")
    if imaging_results['mridone'] == 'MRI Done':
        tests_done.append("MRI")
    if imaging_results['usdone'] == 'US Done':
        tests_done.append("ultrasound")
    
    if tests_done:
        explanation.append(f"• The {', '.join(tests_done)} helped us see {info['typical_findings']}")
    
    return "\n".join(explanation)

def get_treatment_explanation(treatment_type, stage, hiv_status):
    """Generate personalized treatment explanation based on patient factors"""
    if treatment_type not in TREATMENT_DESCRIPTIONS:
        return "The treatment information is not available."
    
    treatment = TREATMENT_DESCRIPTIONS[treatment_type]
    stage_level = "advanced_stage" if int(stage.split(':')[0].split()[-1]) >= 4 else "early_stage"
    
    explanation = [
        f"\n💊 About Your Treatment Plan: {treatment['name']}",
        f"• What it is: {treatment['explanation']}",
        f"• Why this treatment: {treatment['best_for'][stage_level]}"
    ]
    
    if hiv_status == "HIV Positive":
        explanation.append(f"• Special considerations: {treatment['best_for']['with_hiv']}")
    
    return "\n".join(explanation)

def generate_detailed_explanation(row):
    """Generate a focused explanation covering only cancer type, stage, and treatment"""
    explanation = []
    
    # Explain cancer type
    cancer_type = row['biopsyhisto']
    if cancer_type in MEDICAL_TERMS:
        explanation.append(f"🔬 Your Type of Cancer:")
        explanation.append(f"• {MEDICAL_TERMS[cancer_type]['explanation']}")
    
    # Add staging explanation
    explanation.append(get_stage_explanation(row['stage'], {
        'cxrdone': row['cxrdone'],
        'ctdone': row['ctdone'],
        'mridone': row['mridone'],
        'usdone': row['usdone']
    }))
    
    # Add treatment explanation
    explanation.append(get_treatment_explanation(row['txtype'], row['stage'], row['hiv']))
    
    return "\n".join(explanation)

def main():
    st.title("Friendly Medical Report Explainer")
    st.write("This tool explains medical reports in simple terms that anyone can understand! 🏥")

    # Patient Data Inputs
    age = st.number_input("Patient's age:", min_value=0, max_value=100, value=50)
    
    biopsyhisto = st.selectbox(
        "Type of cancer found in biopsy:", 
        options=["Adenocarcinoma", "Squamous Cell Carcinoma", "Unknown Biopsy Type"],
        help="A biopsy is when doctors take a tiny sample of tissue to check for cancer"
    )
    
    stage = st.selectbox(
        "Cancer Stage:", 
        options=["Stage 2: Early localized cancer", "Stage 3: Advanced localized cancer", 
                "Stage 4: Regional spread of cancer", "Stage 6: Extensive cancer spread",
                "Stage 7: Widespread metastasis", "Stage 8: Critical condition", 
                "Stage 9: End-stage cancer", "Unknown Stage"],
        help="The stage tells us how far the cancer has spread in the body"
    )
    
    hiv_status = st.selectbox(
        "HIV Status:", 
        options=["HIV Positive", "HIV Negative", "Unknown HIV Status"],
        help="HIV affects the body's immune system (our natural defense against illness)"
    )
    
    # Imaging tests with explanations
    st.subheader("Imaging Tests Done")
    col1, col2 = st.columns(2)
    
    with col1:
        cxrdone = st.selectbox("Chest X-ray:", options=["CXR Done", "CXR Not Done", "Unknown CXR Status"])
        usdone = st.selectbox("Ultrasound:", options=["US Done", "US Not Done", "Unknown US Status"])
    
    with col2:
        ctdone = st.selectbox("CT Scan:", options=["CT Done", "CT Not Done", "Unknown CT Status"])
        mridone = st.selectbox("MRI Scan:", options=["MRI Done", "MRI Not Done", "Unknown MRI Status"])
    
    txtype = st.selectbox(
        "Treatment Type:", 
        options=["Chemotherapy", "Radiation Therapy", "Immunotherapy", 
                "Hormone Therapy", "Targeted Therapy", "Unknown Treatment Type"],
        help="Different types of treatment work in different ways to fight cancer"
    )

    # Create DataFrame from inputs
    data = pd.DataFrame({
        'age': [age],
        'biopsyhisto': [biopsyhisto],
        'stage': [stage],
        'hiv': [hiv_status],
        'cxrdone': [cxrdone],
        'usdone': [usdone],
        'ctdone': [ctdone],
        'mridone': [mridone],
        'txtype': [txtype]
    })
    
    if st.button("Explain This Report"):
        st.subheader("📋 Your Personalized Medical Report Explanation")
        explanation = generate_detailed_explanation(data.iloc[0])
        st.markdown(explanation)
        
        # Additional helpful resources
        with st.expander("🔍 Want to Learn More About Medical Terms?"):
            st.write("Click here to see explanations of medical terms we used:")
            for term, info in MEDICAL_TERMS.items():
                st.write(f"**{term}**: {info['explanation']}")

if __name__ == "__main__":
    main()