!pip install presidio-analyzer presidio-anonymizer

import logging
from typing import List, Dict, Optional

# Import Microsoft Presidio classes
# Presidio provides the architectural scaffolding for PII detection
from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    RecognizerRegistry,
    RecognizerResult
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configure logging for audit trails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurePromptSanitizer:
    """
A robust 'Sanitizer' pattern implementation for LLM prompts.
It acts as a proxy filter to detect and redact PII and Secrets before
the prompt is sent to an external model.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Initialize the Analyzer and Anonymizer engines.

        Args:
            confidence_threshold (float): The minimum confidence score (0-1) required
                                          to consider a detection valid.
        """
        self.confidence_threshold = confidence_threshold

        # 1. Initialize the Registry. This loads the default Spacy-based NER models
        # for detecting entities like PERSON, LOCATION, DATE.
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        # 2. Add Custom Recognizers for Secrets (not covered by default PII models)
        # This is crucial for preventing "Secret Sprawl" in developer prompts.
        self._add_custom_secret_recognizers(registry)

        # 3. Instantiate the Analyzer Engine with the enriched registry
        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()

        logger.info("SecurePromptSanitizer initialized with custom secret detectors.")

    def _add_custom_secret_recognizers(self, registry: RecognizerRegistry):
        """
        Registers custom Regex patterns for API Keys and Secrets.
        Reference: Patterns derived from AWS documentation and security research.
        """

        # Pattern for AWS Access Key ID (Standard format: AKIA...) 
        # Regex explanation:
        # (?<![A-Z0-9]) : Negative lookbehind to ensure it's a distinct token (boundary check)
        # (AKIA|ASIA|AROA)[A-Z0-9]{16} : The AWS prefix + 16 alphanumeric chars
        # (?![A-Z0-9]) : Negative lookahead
        aws_key_pattern = Pattern(
            name="aws_access_key_pattern",
            regex=r"(?<![A-Z0-9])(AKIA|ASIA|AROA)[A-Z0-9]{16}(?![A-Z0-9])",
            score=1.0  # High confidence for this specific, low-collision pattern
        )

        aws_recognizer = PatternRecognizer(
            supported_entity="AWS_ACCESS_KEY",
            patterns=[aws_key_pattern],
            context=["aws", "key", "id", "access", "secret"] # Context words boost score via ContextAwareEnhancer
        )

        # Pattern for Generic API Keys (High Entropy Hex/Alphanumeric)
        # This relies on context (e.g. "api_key =") to avoid false positives.
        generic_key_pattern = Pattern(
            name="generic_api_key_pattern",
            regex=r"(?i)(api_key|apikey|secret|token)\s*[:=]\s*['\"]?([a-zA-Z0-9]{32,45})['\"]?",
            score=0.6 # Lower base score, relies on context to confirm
        )

        api_key_recognizer = PatternRecognizer(
            supported_entity="GENERIC_API_KEY",
            patterns=[generic_key_pattern]
        )

        registry.add_recognizer(aws_recognizer)
        registry.add_recognizer(api_key_recognizer)

    def sanitize_prompt(self, prompt_text: str) -> Dict:
        """
        Scans the input prompt for sensitive data and returns the sanitized version.

        Args:
            prompt_text (str): The user's raw input prompt.

        Returns:
            Dict: Contains 'original_text', 'sanitized_text', and 'detected_items'.
        """
        if not prompt_text:
            return {"sanitized_text": "", "detected_items": []}

        # Step 1: Analyze the text to find entities
        # This triggers the NER models and Regex patterns in parallel.
        results = self.analyzer.analyze(
            text=prompt_text,
            language='en',
            score_threshold=self.confidence_threshold
        )

        # Step 2: Anonymize (Redact) the detected entities
        # We use 'replace' operator to substitute with <ENTITY_TYPE>.
        # This maintains the prompt's semantic utility.
        anonymized_result = self.anonymizer.anonymize(
            text=prompt_text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<SENSITIVE_DATA>"}),
                "AWS_ACCESS_KEY": OperatorConfig("replace", {"new_value": "<AWS_KEY>"}),
                "GENERIC_API_KEY": OperatorConfig("replace", {"new_value": "<API_KEY>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "US_SSN": OperatorConfig("replace", {"new_value": "<US_SSN>"}),
            }
        )

        # Step 3: Structure the output log for auditing
        detected_items = [
            {
                "type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score
            }
            for res in results
        ]

        logger.info(f"Sanitization complete. Detected {len(detected_items)} sensitive entities.")

        return {
            "original_text": prompt_text,
            "sanitized_text": anonymized_result.text,
            "detected_items": detected_items
        }

# --- Execution Example ---
if __name__ == "__main__":
    # Example Prompt containing PII and Secrets
    # Simulates a developer pasting code + context into an LLM
    user_prompt = """
    Hi AI, I need you to debug this connection string.
    My email is johndoe@example.com and my phone is 555-0199.
    I'm using the AWS key AKIAIOSFODNN7EXAMPLE to connect.
    Also, here is my API token: api_key = "abc1234567890abcdef1234567890abc"
    """

    # Initialize the Security Filter
    sanitizer = SecurePromptSanitizer(confidence_threshold=0.4)

    # Run the Sanitizer
    result = sanitizer.sanitize_prompt(user_prompt)

    # Output Results
    print("\n--- Project 8: Secure Prompting Results ---")
    print(f"Original Prompt Length: {len(result['original_text'])}")
    print(f"\n:\n{result['sanitized_text']}")
    print(f"\n:")
    for item in result['detected_items']:
        print(f"- Detected {item['type']} with confidence {item['score']:.2f}")