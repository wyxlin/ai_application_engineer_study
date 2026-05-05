from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Chain 1 - extract JD requirements
class JDRequirements(BaseModel):
    requirements_skills: list[str]
    experience_years: int
    
jd_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract skills and experience from this job description. Return structured data."),
    ("user", "<job_description>{job_description}</job_description>")
])

jd_chain = jd_prompt | llm.with_structured_output(JDRequirements)

# Chain 2 - extract resume skills
class ResumeProfile(BaseModel):
    skills: list[str]
    experience_years: int
    
resume_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract skills and experience from this resume. Return structured data."),
    ("user", "<resume>{resume}</resume>")
])

resume_chain = resume_prompt | llm.with_structured_output(ResumeProfile)

# Chain 3 - evaluate fit
class FitEvaluation(BaseModel):
    fit_score: int
    reasoning: str
    strengths: list[str]
    gaps: list[str]
    
fit_prompt = ChatPromptTemplate.from_messages([
    ("system", "Evaluate the fit between the JD requirements and the resume profile. Return structured data."),
    ("user", "<jd_requirements>{jd_requirements}</jd_requirements>\n\n<resume_profile>{resume_profile}</resume_profile>")
])

fit_chain = fit_prompt | llm.with_structured_output(FitEvaluation)

def evaluate_pipeline(job_description: str, resume: str) -> FitEvaluation:
    jd_requirements = jd_chain.invoke({"job_description": job_description})
    resume_profile = resume_chain.invoke({"resume": resume})
    fit_evaluation = fit_chain.invoke({"jd_requirements": jd_requirements, "resume_profile": resume_profile})
    return fit_evaluation

# run 
pipeline_result = evaluate_pipeline(
    job_description="Backend Engineer at Stripe: Python, REST APIs, SQL, distributed systems.",
    resume="Lin He: Python, gRPC, distributed systems, SQL, gRPC banking system project"
)

print(pipeline_result)