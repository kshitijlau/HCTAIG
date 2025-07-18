import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="HCTA AI Report Generator",
    page_icon="🤖",
    layout="wide"
)

# --- Savant Prompt Template (Final, Untrimmed Version) ---
# This is the complete, untrimmed prompt with the full dictionary and all examples.
SAVANT_PROMPT_TEMPLATE = """
# Gemini, ACT as an expert-level talent assessment analyst and report writer. Your name is "AnalystAI".
# Your task is to generate a concise, insightful, and professional leadership potential summary based on candidate data.
# You must adhere to all rules, formats, and interpretation logic provided below without deviation.

# --- ABSOLUTE RULES & WRITING STYLE ---
# 1.  **Tone & Language:** Write in the third person, present tense only. Use professional, neutral language and American English spelling. Avoid judgmental, speculative, or robotic tones.
# 2.  **Word Count:** The entire summary paragraph must be under 200 words.
# 3.  **Anonymity:** Do not mention AI, tools, processes, or the names of the assessments.
# 4.  **Behavioral Framing:** All bullet points must be one sentence each and framed in behavioral terms. Do not name competencies directly.
# 5.  **No Ratings:** Do not mention numeric scores or use rating-like terms. Use only the provided behavioral interpretations from the dictionary.
# 6.  **Constructive Language:** Avoid value-laden terms like "good," "bad," or "lacks." Instead, use phrases like "…may enhance impact by…", "…has an opportunity to develop…", or "…demonstrates moderate capability in…".
# 7.  **Pronouns:** Use pronouns (he/she, his/her) that match the provided `Gender` input (M/F).

# --- FORMAT & STRUCTURE (NON-NEGOTIABLE) ---
# 1.  **One-Paragraph Summary:**
#     - Start the paragraph *exactly* with the text from the "Overall Leadership" interpretation. For example: "John demonstrates moderate potential with a reasonable capacity for growth..."
#     - Describe the candidate's likely workplace behaviors based on the provided score interpretations from the dictionary.
#     - Synthesize patterns across all competencies, using the provided mapping for BS and TI. Focus on standout strengths and development areas.
# 2.  **Bullet Points (Strengths & Development Areas):**
#     - After the paragraph, provide exactly two strengths and two development areas.
#     - Use the heading "Strengths:" and "Development Areas:".
#     - These points must extend or complement the paragraph, not repeat it.

# --- LOGIC & INTERPRETATION ENGINE ---
# 1.  **Score Categorization:** High = 3.5-5.0; Moderate = 2.5-3.49; Low = 1.0-2.49.
# 2.  **Strength/Development Rule:** Scores >= 4.0 are *only* strengths. Scores <= 2.0 are *only* development areas.
# 3.  **BS & TI Mapping:**
#     - Steers Changes <-> Change Potential
#     - Manages Stakeholders <-> People Potential
#     - Drives Results <-> Drive Potential
#     - Thinks Strategically <-> Strategic Potential
#     - Solves Challenges <-> Execution Potential
#     - Develops Talent <-> Learning Potential

# --- BEHAVIORAL DICTIONARY (USE THIS TEXT EXACTLY) ---
# **Core Competencies:**
# Overall Leadership:
#   - High: Demonstrates high potential with a strong capacity for growth and success in a more complex role.
#   - Moderate: Demonstrates moderate potential with a reasonable capacity for growth and success in a more complex role.
#   - Low: Demonstrates low potential with a reasonable capacity for growth and success in a more complex role.
# Reasoning & Problem Solving:
#   - High: Candidate demonstrates a higher-than-average reasoning and problem-solving ability as compared to a group of peers.
#   - Moderate: Candidate demonstrates an average reasoning and problem-solving ability as compared to a group of peers.
#   - Low: Candidate demonstrates a below-average reasoning and problem-solving ability as compared to a group of peers.
# **Business Simulation (BS) Competencies:**
# Steers Changes:
#   - High: Strong ability to recognise and drive change and transformation at an organisational level. Displays strong resilience and strength during adversity and is well equipped to enable buy-in and support.
#   - Moderate: Moderate ability to contribute to organisational change and transformation. Shows resilience during challenging times and can occasionally support others in gaining buy-in.
#   - Low: Limited ability to support change and transformation at an organisational level. Struggles to remain resilient during adversity and has difficulty enabling buy-in and support.
# Manages Stakeholders:
#   - High: Strong ability to develop and nurture relationships with key stakeholders. Actively finds synergies between organisations to ensure positive outcomes. Networks with stakeholders within and outside one’s industry to stay up-to-date about new developments.
#   - Moderate: Moderate ability to maintain and build relationships with key stakeholders. Occasionally identifies synergies between organisations and engages with stakeholders to stay informed of developments.
#   - Low: Limited ability to develop and maintain relationships with stakeholders. Rarely identifies synergies between organisations or engages with external stakeholders to stay informed.
# Drives Results:
#   - High: Strong ability to articulate performance standards and metrics that support the achievement of organisational goals. Ensures a high-performance culture across teams and demonstrates grit in achievement of challenging goals.
#   - Moderate: Moderate ability to articulate performance standards and metrics that contribute to achieving organisational goals. Occasionally supports performance across teams and shows persistence when working towards goals.
#   - Low: Low ability to articulate performance standards and metrics that support organisational goals. Needs development in fostering a high-performance culture and in maintaining persistence when faced with challenging goals.
# Thinks Strategically:
#   - High: Strong ability to balance the achievement of short-term results with creating long-term value and competitive advantage. Successfully translates complex organisational goals into meaningful actions across teams.
#   - Moderate: Moderate ability to balance short-term results with long-term priorities. Occasionally translates organisational goals into meaningful actions across teams.
#   - Low: Low ability to balance short-term performance with long-term value creation. Struggles to translate organisational goals into meaningful team actions.
# Solves Challenges:
#   - High: Strong ability to deal with ambiguous and complex situations, by making tough decisions where necessary. Is comfortable leading in an environment where goals are frequently complex and thrives during periods of uncertainty.
#   - Moderate: Moderate ability to handle some ambiguous and complex situations by making necessary decisions. Shows some confidence in leading through moderately uncertain environments.
#   - Low: Low ability to deal with ambiguity and complexity. Hesitant to make tough decisions and limited confidence in leading through uncertain situations.
# Develops Talent:
#   - High: Strong ability to leverage and nurture individual strengths to achieve positive outcomes. Actively fosters a culture of learning and advocates for career advancement opportunities within the organisation.
#   - Moderate: Moderate ability to recognise and utilise individual strengths to support positive outcomes. Supports learning and contributes to career development within the organisation.
#   - Low: Low ability to identify and leverage individual strengths. Rarely supports learning or advocates for career development within the organisation.
# **Thriving Index (TI) Potentials & Factors:**
# Drive Potential:
#   - High: Consistently demonstrates a positive mindset and motivation; regularly takes initiative to exceed expectations with a strong drive to achieve goals, targets, and results. Seeks fulfillment through impact.
#   - Moderate: Shows a generally positive mindset and some motivation; occasionally takes initiative and shows a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained.
#   - Low: Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident.
# Learning Potential:
#   - High: Consistently takes time to focus on personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn.
#   - Moderate: Shows some effort toward personal and professional growth. Engages in learning activities but may not do so consistently. Some openness to learning and unlearning.
#   - Low: Rarely focuses on personal or professional growth. Engagement in learning is limited and may resist feedback or change.
# People Potential:
#   - High: Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interactions.
#   - Moderate: Displays some ability to relate to and lead others. May show empathy and focus on people inconsistently. Builds relationships but may need support.
#   - Low: Shows limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships.
# Strategic Potential:
#   - High: Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact.
#   - Moderate: Some awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications.
#   - Low: Focus tends to be on immediate tasks. Requires frequent guidance. Shows limited awareness of trends or the strategic impact of work.
# Execution Potential:
#   - High: Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach to solving issues.
#   - Moderate: Can address problems but may need support or time to build confidence and resilience. Attempts a practical approach but not always solution-focused.
#   - Low: Struggles to address problems confidently. May rely heavily on others. Practical or solution-oriented approaches are limited.
# Change Potential:
#   - High: Thrives in change and complexity. Manages new ways of working with adaptability, flexibility, and decisiveness during uncertainty.
#   - Moderate: Generally copes with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations.
#   - Low: Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances.

# --- GOLD STANDARD EXAMPLES (LEARN FROM THESE) ---
# **EXAMPLE 1:**
# **INPUT:** Name: Sub 1, Gender: M, Overall Leadership: 4, Reasoning & Problem Solving: 4, Drive Potential: 4, Contribution: 5, Purpose: 4, Achievement: 2, Learning Potential: 3, Mastery: 3, Growth: 3, Insightful: 3, People Potential: 4, Collaboration: 4, Empathy: 4, Sociable: 5, Strategic Potential: 4, Awareness: 5, Autonomy: 3, Perspective: 4, Execution Potential: 5, Resourcefulness: 5, Efficacy: 5, Resilience: 5, Change Potential: 4, Agility: 5, Ambiguity: 5, Venturesome: 3, Steers Changes: 5, Manages Stakeholders: 4, Drives Results: 5, Thinks Strategically: 4, Solves Challenges: 5, Develops Talent: 3
# **CORRECT OUTPUT:**
# Sub1 demonstrates high leadership potential and the ability to operate effectively in increasingly complex roles. He is driven, resilient, and purpose-oriented, consistently exceeding expectations while maintaining a learning mindset. His ability to inspire others, collaborate across boundaries, and display emotional intelligence in team dynamics stands out. He shows high ownership of his development, with a solid grasp of goal alignment and delivery under pressure. While highly sociable and strategically aware, the candidate’s ability to handle ambiguity and data interpretation is still maturing. He is comfortable with change, taking initiative, and influencing outcomes proactively. Continued focus on building strategic insight and sharpening analytical depth will help him transition to higher-impact roles more seamlessly.
#
# Strengths:
# • Demonstrates drive and resilience, consistently going beyond expectations while maintaining focus on outcomes.
# • High sociability and collaboration; effectively leads and engages others across teams with strong interpersonal impact.
#
# Development Areas:
# • May benefit from actively seeking learning opportunities and showing openness to new ways of thinking.
# • Has an opportunity to strengthen leadership impact by investing more in supporting the development of others.

# **EXAMPLE 2:**
# **INPUT:** Name: John Doe, Gender: M, Overall Leadership: 3, Reasoning & Problem Solving: 3, Drive Potential: 2, Contribution: 2, Purpose: 2, Achievement: 1, Learning Potential: 2, Mastery: 1, Growth: 3, Insightful: 2, People Potential: 3, Collaboration: 3, Empathy: 3, Sociable: 4, Strategic Potential: 3, Awareness: 3, Autonomy: 3, Perspective: 3, Execution Potential: 3, Resourcefulness: 3, Efficacy: 3, Resilience: 3, Change Potential: 2, Agility: 3, Ambiguity: 3, Venturesome: 3, Steers Changes: 2, Manages Stakeholders: 3, Drives Results: 1, Thinks Strategically: 2, Solves Challenges: 3, Develops Talent: 1
# **CORRECT OUTPUT:**
# John demonstrates moderate leadership potential, with strengths in resilience and collaborative behaviors. He shows the ability to stay composed under pressure and contributes positively to team settings. His responses suggest a practical mindset and the ability to support group goals, especially in stable or familiar contexts. However, he may benefit from taking more initiative, particularly in unstructured or high-accountability situations. His approach to learning appears more reactive than proactive, and he may not consistently seek opportunities to expand his skillset. The ability to develop others also appears limited, indicating an opportunity to more actively support and grow talent around him. Enhancing learning agility and ownership could help him elevate his overall leadership impact.
#
# Strengths:
# • Maintains a calm and solution-oriented approach under pressure, supporting consistent delivery.
# • Builds constructive team relationships and collaborates effectively to meet shared goals.
#
# Development Areas:
# • May benefit from proactively seeking learning opportunities to build broader adaptability and ownership, particularly in ambiguous situations.
# • Limited strategic clarity and learning orientation restrict consistent performance elevation.

# **EXAMPLE 3:**
# **INPUT:** Name: Jane Doe, Gender: F, Overall Leadership: 2, Reasoning & Problem Solving: 1, Drive Potential: 3, Contribution: 2, Purpose: 3, Achievement: 4, Learning Potential: 4, Mastery: 4, Growth: 5, Insightful: 3, People Potential: 2, Collaboration: 3, Empathy: 2, Sociable: 1, Strategic Potential: 2, Awareness: 1, Autonomy: 2, Perspective: 3, Execution Potential: 2, Resourcefulness: 2, Efficacy: 1, Resilience: 2, Change Potential: 2, Agility: 1, Ambiguity: 1, Venturesome: 3, Steers Changes: 1, Manages Stakeholders: 1, Drives Results: 2, Thinks Strategically: 1, Solves Challenges: 2, Develops Talent: 4
# **CORRECT OUTPUT:**
# Jane demonstrates moderate leadership potential, with emerging strengths in resilience and team collaboration. She generally maintains a constructive mindset and engages well in group settings, particularly when expectations are clearly defined. Her responses suggest that she benefits from external structure and guidance, which can support her contribution in routine or familiar situations. However, she may be less confident when required to act independently, particularly in ambiguous or high-responsibility contexts. Strategic orientation and clarity of purpose also appear limited, which may affect her ability to take initiative or contribute meaningfully to longer-term goals. With targeted support to build autonomy and forward-thinking behaviors, Jane can continue strengthening her readiness for broader leadership responsibility.
#
# Strengths:
# • Demonstrates a generally positive mindset and can collaborate effectively when provided with direction.
# • Shows moderate resilience and willingness to recover from setbacks with some support.
#
# Development Areas:
# • Needs to build independence and initiative; currently depends too much on guidance to perform consistently.
# • Lacks clarity in purpose and strategic thinking, limiting the ability to contribute meaningfully to complex goals.

# **EXAMPLE 4:**
# **INPUT:** Name: Anvita Sirohi, Gender: F, Overall Leadership: 3, Reasoning & Problem Solving: 4, Drive Potential: 4, Contribution: 4, Purpose: 4, Achievement: 1, Learning Potential: 3, Mastery: 3, Growth: 3, Insightful: 3, People Potential: 5, Collaboration: 5, Empathy: 5, Sociable: 4, Strategic Potential: 4, Awareness: 4, Autonomy: 3, Perspective: 5, Execution Potential: 4, Resourcefulness: 4, Efficacy: 4, Resilience: 5, Change Potential: 4, Agility: 4, Ambiguity: 5, Venturesome: 3, Steers Changes: 3, Manages Stakeholders: 2, Drives Results: 2, Thinks Strategically: 2, Solves Challenges: 3, Develops Talent: 4
# **CORRECT OUTPUT:**
# Anvita Sirohi demonstrates moderate leadership potential, with strengths in resilience, goal orientation, and consistent personal drive. She tends to stay focused on priorities and shows determination in following through on tasks, even in the face of setbacks. Her ability to maintain confidence and emotional stability supports steady execution and a results-oriented mindset. She demonstrates a generally independent working style, occasionally drawing on external input when needed. While her capacity to adapt to change is evident, she may benefit from developing more comfort with navigating uncertainty or shifting priorities. There is also room to broaden her strategic awareness and deepen stakeholder engagement to enhance her broader leadership impact.
#
# Strengths:
# • Remains goal-focused and shows commitment to follow-through, even under pressure or after setbacks.
# • Demonstrates resilience and belief in personal capability, contributing to consistent effort and delivery.
#
# Development Areas:
# • May enhance leadership effectiveness by increasing comfort in navigating situations with ambiguity or incomplete information.
# • Has an opportunity to strengthen strategic engagement by deepening awareness of stakeholder needs and the broader impact of decisions.

# **EXAMPLE 5:**
# **INPUT:** Name: Sub 5, Gender: M, Overall Leadership: 4, Reasoning & Problem Solving: 3, Drive Potential: 3, Contribution: 2, Purpose: 3, Achievement: 3, Learning Potential: 4, Mastery: 5, Growth: 5, Insightful: 3, People Potential: 4, Collaboration: 3, Empathy: 2, Sociable: 4, Strategic Potential: 3, Awareness: 2, Autonomy: 3, Perspective: 2, Execution Potential: 4, Resourcefulness: 4, Efficacy: 4, Resilience: 3, Change Potential: 3, Agility: 3, Ambiguity: 2, Venturesome: 3, Steers Changes: 2, Manages Stakeholders: 3, Drives Results: 2, Thinks Strategically: 2, Solves Challenges: 3, Develops Talent: 2
# **CORRECT OUTPUT:**
# Sub5 demonstrates moderate leadership potential, supported by strengths in sociability, collaboration, and emotional resilience. He tends to work well with others, building positive relationships and contributing to group cohesion. His approachable style and willingness to support team efforts allow them to navigate interpersonal dynamics effectively. In challenging situations, he tends to recover quickly and maintain a stable, steady presence. While generally confident and socially comfortable, there is less evidence of proactive goal orientation or strategic follow-through. Subject’s leadership potential may be enhanced by developing greater clarity and discipline in pursuing outcomes, as well as building confidence in decision-making when facing ambiguous or uncertain conditions.
#
# Strengths:
# • Builds rapport with others and helps maintain team cohesion by constructively addressing interpersonal challenges.
# • Demonstrates emotional steadiness and resilience, maintaining performance in the face of setbacks.
#
# Development Areas:
# • May enhance impact by sharpening focus on results and taking greater initiative toward defined outcomes.
# • Has an opportunity to build comfort in making decisions amid uncertainty or when information is incomplete.

# --- END OF INSTRUCTIONS AND EXAMPLES ---

### NEW CANDIDATE DATA TO ANALYZE ###
{candidate_data_string}

# AnalystAI, generate the report now.
"""

# --- App Title and Description ---
st.title("HCTA AI Leadership Potential Report Generator")
st.markdown("""
This application uses the Gemini family of models to generate executive summaries for candidate assessments.
1.  Download the sample template to see the required format and column order.
2.  Fill the template with your candidate data.
3.  Upload the completed Excel file and click "Generate Summaries".
""")

# --- Helper function to create and download the sample Excel file ---
@st.cache_data
def create_sample_template():
    # Define all expected columns in the new, specific order
    columns = [
        'Name', 'Gender', 'Overall Leadership', 'Reasoning & Problem Solving',
        'Drive Potential', 'Contribution', 'Purpose', 'Achievement',
        'Learning Potential', 'Mastery', 'Growth', 'Insightful',
        'People Potential', 'Collaboration', 'Empathy', 'Sociable',
        'Strategic Potential', 'Awareness', 'Autonomy', 'Perspective',
        'Execution Potential', 'Resourcefulness', 'Efficacy', 'Resilience',
        'Change Potential', 'Agility', 'Ambiguity', 'Venturesome',
        'Steers Changes', 'Manages Stakeholders', 'Drives Results',
        'Thinks Strategically', 'Solves Challenges', 'Develops Talent'
    ]
    # Create a sample row
    sample_data = {
        'Name': ['Jane Doe'], 'Gender': ['F'], 'Overall Leadership': [2.0], 'Reasoning & Problem Solving': [1.0],
        'Drive Potential': [3.0], 'Contribution': [2.0], 'Purpose': [3.0], 'Achievement': [4.0],
        'Learning Potential': [4.0], 'Mastery': [4.0], 'Growth': [5.0], 'Insightful': [3.0],
        'People Potential': [2.0], 'Collaboration': [3.0], 'Empathy': [2.0], 'Sociable': [1.0],
        'Strategic Potential': [2.0], 'Awareness': [1.0], 'Autonomy': [2.0], 'Perspective': [3.0],
        'Execution Potential': [2.0], 'Resourcefulness': [2.0], 'Efficacy': [1.0], 'Resilience': [2.0],
        'Change Potential': [2.0], 'Agility': [1.0], 'Ambiguity': [1.0], 'Venturesome': [3.0],
        'Steers Changes': [1.0], 'Manages Stakeholders': [1.0], 'Drives Results': [2.0],
        'Thinks Strategically': [1.0], 'Solves Challenges': [2.0], 'Develops Talent': [4.0]
    }
    df_sample = pd.DataFrame(sample_data, columns=columns)
    
    # Convert to Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='Candidates')
    processed_data = output.getvalue()
    return processed_data

sample_excel = create_sample_template()
st.download_button(
    label="📥 Download Sample Template (Excel)",
    data=sample_excel,
    file_name="candidate_scores_template_v3.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload Your Completed Excel File", type=["xlsx"])

# --- Main Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File Uploaded Successfully!")
        st.dataframe(df)

        if st.button("🚀 Generate Summaries", type="primary"):
            # Check for API Key
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
            except (KeyError, FileNotFoundError):
                st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
                st.stop()
            
            # Using Gemini 1.5 Pro based on the user's initial request
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            results = []
            total_candidates = len(df)
            progress_bar = st.progress(0, text="Initializing...")

            # Use a container for results to appear as they are generated
            results_container = st.container()

            for index, row in df.iterrows():
                progress_text = f"Generating summary for {row['Name']} ({index + 1}/{total_candidates})..."
                progress_bar.progress((index + 1) / total_candidates, text=progress_text)
                
                # Create the data string for the prompt
                candidate_data_string = "# INPUT SCORES:\n"
                for col_name, value in row.items():
                    candidate_data_string += f"# {col_name}: {value}\n"

                # Format the final prompt
                final_prompt = SAVANT_PROMPT_TEMPLATE.format(candidate_data_string=candidate_data_string)
                
                try:
                    # API Call
                    response = model.generate_content(final_prompt)
                    summary = response.text
                except Exception as e:
                    summary = f"Error generating summary: {e}"
                
                results.append(summary)
                
                # Display result immediately
                with results_container:
                    st.subheader(f"Summary for {row['Name']}")
                    st.markdown(summary)
                    st.divider()

                # A small delay can help prevent hitting API rate limits on rapid, large batches.
                time.sleep(2) 

            progress_bar.empty()
            st.success("✅ All summaries generated!")
            
            df['Generated Summary'] = results
            
            # --- Download Results ---
            output_results = io.BytesIO()
            with pd.ExcelWriter(output_results, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            
            st.download_button(
                label="⬇️ Download All Results with Summaries (Excel)",
                data=output_results.getvalue(),
                file_name="candidate_summaries_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

