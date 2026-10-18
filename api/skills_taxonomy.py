"""Canonical skill names and the aliases that map to them.

Used by both metadata extraction (resumes) and JD parsing so that
"ML", "machine-learning" and "Machine Learning" all land on one skill.
"""
import re

SKILL_ALIASES = {
    # Languages
    "Python": ["python", "python3"],
    "Java": ["java", "java 8", "java 11", "java 17"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript"],
    "Go": ["golang", "go lang"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Scala": ["scala"],
    "SQL": ["sql", "t-sql", "pl/sql"],
    "R": ["r programming", "r language"],
    "C++": ["c++", "cpp"],
    "Bash": ["bash", "shell scripting"],
    # Backend
    "Spring Boot": ["spring boot", "springboot"],
    "Hibernate": ["hibernate", "jpa"],
    "Microservices": ["microservices", "micro-services", "microservice"],
    "REST APIs": ["rest api", "rest apis", "restful"],
    "GraphQL": ["graphql"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Node.js": ["node.js", "nodejs"],
    "Kafka": ["kafka", "apache kafka"],
    "Redis": ["redis"],
    "RabbitMQ": ["rabbitmq"],
    # Data / ML
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural networks"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv models"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "LLMs": ["llm", "llms", "large language models", "gpt", "transformers"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "MLOps": ["mlops", "mlflow", "kubeflow"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "Hadoop": ["hadoop", "hdfs", "hive"],
    "ETL": ["etl", "elt", "data pipelines", "data pipeline"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Excel": ["excel", "advanced excel"],
    "Statistics": ["statistics", "statistical analysis", "a/b testing", "hypothesis testing"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "DynamoDB": ["dynamodb"],
    # Cloud / DevOps
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud", "bigquery"],
    "Docker": ["docker", "containers"],
    "Kubernetes": ["kubernetes", "k8s", "eks", "gke"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "CI/CD": ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci"],
    "Prometheus": ["prometheus", "grafana"],
    "Linux": ["linux", "unix"],
    # Frontend / mobile
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Redux": ["redux"],
    "HTML/CSS": ["html", "css", "html5", "css3", "tailwind", "sass"],
    "Android": ["android"],
    "iOS": ["ios"],
    "React Native": ["react native"],
    "Flutter": ["flutter", "dart"],
    # QA
    "Selenium": ["selenium"],
    "Cypress": ["cypress"],
    "Test Automation": ["test automation", "automation testing", "automated testing"],
    "JUnit": ["junit", "testng"],
    "Pytest": ["pytest"],
    "Postman": ["postman", "api testing"],
    # Practices
    "System Design": ["system design", "distributed systems"],
    "Agile": ["agile", "scrum"],
}

# Aliases that are short / ambiguous and need exact-case or word-boundary care
_CASE_SENSITIVE = {"R": ["R"], "Go": ["Go"]}

_PATTERNS = []
for canon, aliases in SKILL_ALIASES.items():
    for a in sorted(set(aliases + [canon.lower()]), key=len, reverse=True):
        if canon in ("R", "Go") and a in ("r", "go"):
            continue  # handled case-sensitively below
        pat = re.compile(r"(?<![A-Za-z0-9+#])" + re.escape(a) + r"(?![A-Za-z0-9+#])", re.I)
        _PATTERNS.append((canon, pat))
for canon, toks in _CASE_SENSITIVE.items():
    for t in toks:
        _PATTERNS.append((canon, re.compile(r"(?<![A-Za-z0-9+#/])" + re.escape(t) + r"(?![A-Za-z0-9+#/.'])")))


def extract_skills(text: str) -> list[str]:
    """Return canonical skills mentioned in text, in order of first appearance."""
    found = {}
    for canon, pat in _PATTERNS:
        m = pat.search(text)
        if m and (canon not in found or m.start() < found[canon]):
            found[canon] = m.start()
    return [s for s, _ in sorted(found.items(), key=lambda kv: kv[1])]


def skill_key(skill: str) -> str:
    """Metadata-safe key, e.g. 'Spring Boot' -> 'skill_spring_boot'."""
    return "skill_" + re.sub(r"[^a-z0-9]+", "_", skill.lower()).strip("_")


def skill_patterns(skill: str) -> list[str]:
    """All surface forms of a canonical skill (for keyword search / highlighting)."""
    return sorted(set([skill.lower()] + SKILL_ALIASES.get(skill, [])), key=len, reverse=True)
