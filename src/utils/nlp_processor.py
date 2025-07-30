"""
Natural Language Processing Utilities for Personal Data Analysis

This module provides comprehensive NLP capabilities for analyzing personal data:
- Text classification and categorization
- Entity extraction and recognition
- Topic modeling and theme identification
- Sentiment analysis
- Language detection
- Text summarization
- Privacy-aware text processing
"""

import re
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import textstat
import structlog

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class TextCategory(str, Enum):
    """Text classification categories"""
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    FINANCIAL = "financial" 
    HEALTH = "health"
    TRAVEL = "travel"
    EDUCATION = "education"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SPAM = "spam"
    OTHER = "other"


class UrgencyLevel(str, Enum):
    """Text urgency classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class EntityMention:
    """Represents a named entity mention in text"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    canonical_form: Optional[str] = None


@dataclass
class TextAnalysisResult:
    """Comprehensive text analysis results"""
    text: str
    language: str
    sentiment_score: float
    sentiment_label: SentimentLabel
    urgency_level: UrgencyLevel
    category: TextCategory
    entities: List[EntityMention]
    topics: List[Tuple[str, float]]
    keywords: List[Tuple[str, float]]
    readability_score: float
    word_count: int
    sentence_count: int
    processed_at: datetime


@dataclass
class TopicModel:
    """Represents a trained topic model"""
    model_type: str  # 'lda' or 'nmf'
    n_topics: int
    topics: List[List[Tuple[str, float]]]
    model: Any
    vectorizer: Any
    created_at: datetime


class NLPProcessor:
    """Main NLP processing class with privacy-aware features"""
    
    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        self._models = {}
        self._sentiment_analyzer = None
        self._spacy_nlp = None
        self._transformers_pipelines = {}
        self._topic_models = {}
        self._text_classifier = None
        self._urgency_classifier = None
        
        # Privacy settings
        self._anonymize_entities = True
        self._entity_hash_salt = self.settings.encryption.key_salt.get_secret_value()
    
    async def initialize(self):
        """Initialize NLP models and resources"""
        if self._initialized:
            return
        
        logger.info("Initializing NLP processor")
        
        # Download required NLTK data
        await self._download_nltk_data()
        
        # Initialize sentiment analyzer
        self._sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, some features will be limited")
        
        # Initialize transformers if available
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            await self._initialize_transformers()
        
        # Load or train classification models
        await self._initialize_classifiers()
        
        self._initialized = True
        logger.info("NLP processor initialized successfully")
    
    async def _download_nltk_data(self):
        """Download required NLTK datasets"""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'vader_lexicon',
            'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
        ]
        
        for dataset in required_data:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
    
    async def _initialize_transformers(self):
        """Initialize transformer-based models"""
        try:
            # Initialize sentiment analysis pipeline
            self._transformers_pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Initialize text classification pipeline
            self._transformers_pipelines['classification'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            logger.info("Transformer models initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize transformers: {e}")
    
    async def _initialize_classifiers(self):
        """Initialize or load text classifiers"""
        # For now, use rule-based classification
        # In production, these would be trained on labeled data
        self._urgency_keywords = {
            UrgencyLevel.URGENT: [
                'urgent', 'asap', 'immediately', 'emergency', 'critical',
                'deadline', 'rush', 'important', '!!!', 'help'
            ],
            UrgencyLevel.HIGH: [
                'important', 'priority', 'needed', 'required', 'soon',
                'today', 'tomorrow', 'meeting', 'call me'
            ],
            UrgencyLevel.MEDIUM: [
                'please', 'could you', 'when possible', 'update',
                'follow up', 'reminder', 'check'
            ]
        }
        
        self._category_keywords = {
            TextCategory.PROFESSIONAL: [
                'meeting', 'project', 'deadline', 'client', 'proposal',
                'budget', 'report', 'presentation', 'contract', 'team'
            ],
            TextCategory.FINANCIAL: [
                'payment', 'invoice', 'bank', 'money', 'dollar', 'cost',
                'price', 'budget', 'transaction', 'account', 'credit'
            ],
            TextCategory.HEALTH: [
                'doctor', 'appointment', 'medical', 'health', 'prescription',
                'hospital', 'clinic', 'insurance', 'symptom', 'treatment'
            ],
            TextCategory.TRAVEL: [
                'flight', 'hotel', 'travel', 'trip', 'vacation', 'booking',
                'airport', 'passport', 'visa', 'itinerary'
            ]
        }
    
    async def analyze_text(
        self,
        text: str,
        include_entities: bool = True,
        include_topics: bool = False,
        anonymize: bool = None
    ) -> TextAnalysisResult:
        """Perform comprehensive text analysis"""
        if not self._initialized:
            await self.initialize()
        
        if anonymize is None:
            anonymize = self._anonymize_entities
        
        # Basic text statistics
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        
        # Language detection (simplified)
        language = self._detect_language(text)
        
        # Sentiment analysis
        sentiment_score, sentiment_label = await self._analyze_sentiment(text)
        
        # Urgency classification
        urgency_level = self._classify_urgency(text)
        
        # Category classification
        category = self._classify_category(text)
        
        # Entity extraction
        entities = []
        if include_entities:
            entities = await self._extract_entities(text, anonymize=anonymize)
        
        # Topic extraction (if requested)
        topics = []
        if include_topics:
            topics = await self._extract_topics(text)
        
        # Keyword extraction
        keywords = self._extract_keywords(text)
        
        # Readability score
        readability_score = textstat.flesch_reading_ease(text)
        
        return TextAnalysisResult(
            text=text if not anonymize else self._anonymize_text(text),
            language=language,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            urgency_level=urgency_level,
            category=category,
            entities=entities,
            topics=topics,
            keywords=keywords,
            readability_score=readability_score,
            word_count=word_count,
            sentence_count=sentence_count,
            processed_at=datetime.now()
        )
    
    async def batch_analyze_texts(
        self,
        texts: List[str],
        **kwargs
    ) -> List[TextAnalysisResult]:
        """Analyze multiple texts in batch for efficiency"""
        tasks = [self.analyze_text(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper language detection)"""
        # For now, assume English
        # In production, use langdetect or similar
        return "en"
    
    async def _analyze_sentiment(self, text: str) -> Tuple[float, SentimentLabel]:
        """Analyze sentiment using multiple methods"""
        # NLTK VADER sentiment
        scores = self._sentiment_analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        # Use transformers if available for better accuracy
        if 'sentiment' in self._transformers_pipelines:
            try:
                result = self._transformers_pipelines['sentiment'](text)
                # Process transformer results
                transformer_score = self._process_transformer_sentiment(result)
                # Combine scores (weighted average)
                compound_score = 0.3 * compound_score + 0.7 * transformer_score
            except Exception as e:
                logger.warning(f"Transformer sentiment analysis failed: {e}")
        
        # Convert to sentiment label
        if compound_score <= -0.5:
            label = SentimentLabel.VERY_NEGATIVE
        elif compound_score <= -0.1:
            label = SentimentLabel.NEGATIVE
        elif compound_score >= 0.5:
            label = SentimentLabel.VERY_POSITIVE
        elif compound_score >= 0.1:
            label = SentimentLabel.POSITIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        return compound_score, label
    
    def _process_transformer_sentiment(self, result: List[Dict]) -> float:
        """Process transformer sentiment results to compound score"""
        # Convert transformer labels to numeric scores
        score_map = {
            'LABEL_0': -1.0,  # Negative
            'LABEL_1': 0.0,   # Neutral
            'LABEL_2': 1.0,   # Positive
            'NEGATIVE': -1.0,
            'NEUTRAL': 0.0,
            'POSITIVE': 1.0
        }
        
        weighted_score = 0.0
        for item in result:
            label = item['label']
            score = item['score']
            if label in score_map:
                weighted_score += score_map[label] * score
        
        return weighted_score
    
    def _classify_urgency(self, text: str) -> UrgencyLevel:
        """Classify text urgency using keyword matching and patterns"""
        text_lower = text.lower()
        
        # Check for urgent patterns
        urgent_patterns = [
            r'\b(urgent|asap|emergency|critical)\b',
            r'\b(immediate|right now|right away)\b',
            r'!!!+',
            r'\bdeadline.*today\b',
            r'\bhelp.*urgent\b'
        ]
        
        for pattern in urgent_patterns:
            if re.search(pattern, text_lower):
                return UrgencyLevel.URGENT
        
        # Score based on keyword presence
        scores = {level: 0 for level in UrgencyLevel}
        
        for level, keywords in self._urgency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[level] += 1
        
        # Return highest scoring level
        max_level = max(scores.items(), key=lambda x: x[1])
        if max_level[1] > 0:
            return max_level[0]
        
        return UrgencyLevel.LOW
    
    def _classify_category(self, text: str) -> TextCategory:
        """Classify text category using keyword matching"""
        text_lower = text.lower()
        
        scores = {category: 0 for category in TextCategory}
        
        for category, keywords in self._category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        # Check for spam indicators
        spam_patterns = [
            r'\b(win|winner|congratulations|prize)\b.*\b(money|cash|\$)\b',
            r'\b(click here|act now|limited time)\b',
            r'\b(free|urgent|guaranteed)\b.*\b(money|offer)\b'
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                scores[TextCategory.SPAM] += 3
        
        # Return highest scoring category
        max_category = max(scores.items(), key=lambda x: x[1])
        if max_category[1] > 0:
            return max_category[0]
        
        return TextCategory.OTHER
    
    async def _extract_entities(
        self,
        text: str,
        anonymize: bool = True
    ) -> List[EntityMention]:
        """Extract named entities from text"""
        entities = []
        
        # Use spaCy if available
        if self._spacy_nlp:
            doc = self._spacy_nlp(text)
            for ent in doc.ents:
                canonical_form = ent.text
                if anonymize:
                    canonical_form = self._hash_entity(ent.text, ent.label_)
                
                entities.append(EntityMention(
                    text=ent.text if not anonymize else f"[{ent.label_}]",
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores
                    canonical_form=canonical_form
                ))
        else:
            # Fallback to NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags, binary=False)
            
            current_chunk = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):  # Named entity
                    if current_label != chunk.label():
                        if current_chunk:
                            # Process previous chunk
                            entity_text = ' '.join(current_chunk)
                            canonical_form = entity_text
                            if anonymize:
                                canonical_form = self._hash_entity(entity_text, current_label)
                            
                            entities.append(EntityMention(
                                text=entity_text if not anonymize else f"[{current_label}]",
                                label=current_label,
                                start=0,  # NLTK doesn't provide character positions
                                end=0,
                                confidence=0.8,
                                canonical_form=canonical_form
                            ))
                        
                        current_chunk = [chunk[0]]
                        current_label = chunk.label()
                    else:
                        current_chunk.append(chunk[0])
                else:
                    if current_chunk:
                        # End of current entity
                        entity_text = ' '.join(current_chunk)
                        canonical_form = entity_text
                        if anonymize:
                            canonical_form = self._hash_entity(entity_text, current_label)
                        
                        entities.append(EntityMention(
                            text=entity_text if not anonymize else f"[{current_label}]",
                            label=current_label,
                            start=0,
                            end=0,
                            confidence=0.8,
                            canonical_form=canonical_form
                        ))
                        current_chunk = []
                        current_label = None
        
        return entities
    
    def _hash_entity(self, entity_text: str, entity_type: str) -> str:
        """Create anonymized hash for entity"""
        combined = f"{entity_text}_{entity_type}_{self._entity_hash_salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize sensitive information in text"""
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Replace SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Replace credit card numbers
        text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', '[CREDIT_CARD]', text)
        
        return text
    
    async def _extract_topics(self, text: str, n_topics: int = 3) -> List[Tuple[str, float]]:
        """Extract topics from text using simple keyword extraction"""
        # For single document topic extraction, use keyword extraction
        keywords = self._extract_keywords(text, max_keywords=n_topics)
        return [(word, score) for word, score in keywords]
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Filter and lemmatize
        filtered_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words and len(token) > 2
        ]
        
        if not filtered_tokens:
            return []
        
        # Calculate word frequencies
        word_freq = {}
        for token in filtered_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Normalize frequencies
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:max_keywords]
    
    async def train_topic_model(
        self,
        texts: List[str],
        n_topics: int = 10,
        model_type: str = 'lda'
    ) -> TopicModel:
        """Train a topic model on a collection of texts"""
        if not self._initialized:
            await self.initialize()
        
        # Preprocess texts
        processed_texts = []
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in stop_words and len(token) > 2
            ]
            processed_texts.append(' '.join(filtered_tokens))
        
        # Vectorize texts
        if model_type == 'lda':
            vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:  # NMF
            vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            model = NMF(n_components=n_topics, random_state=42)
        
        # Fit model
        model.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_words = topic.argsort()[-10:][::-1]
            topic_words = [
                (feature_names[i], topic[i])
                for i in top_words
            ]
            topics.append(topic_words)
        
        topic_model = TopicModel(
            model_type=model_type,
            n_topics=n_topics,
            topics=topics,
            model=model,
            vectorizer=vectorizer,
            created_at=datetime.now()
        )
        
        # Cache the model
        model_key = f"{model_type}_{n_topics}_{len(texts)}"
        self._topic_models[model_key] = topic_model
        
        return topic_model
    
    async def classify_with_topic_model(
        self,
        text: str,
        topic_model: TopicModel
    ) -> List[Tuple[int, float]]:
        """Classify text using a trained topic model"""
        # Preprocess text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        filtered_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words and len(token) > 2
        ]
        processed_text = ' '.join(filtered_tokens)
        
        # Transform text
        doc_term_matrix = topic_model.vectorizer.transform([processed_text])
        topic_distribution = topic_model.model.transform(doc_term_matrix)[0]
        
        # Return topic probabilities
        return [(i, prob) for i, prob in enumerate(topic_distribution)]
    
    async def summarize_text(
        self,
        text: str,
        max_sentences: int = 3
    ) -> str:
        """Generate extractive text summary"""
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Use TF-IDF to score sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF scores)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences
            top_sentence_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_sentence_indices = sorted(top_sentence_indices)
            
            summary_sentences = [sentences[i] for i in top_sentence_indices]
            return ' '.join(summary_sentences)
            
        except ValueError:
            # Fallback: return first few sentences
            return ' '.join(sentences[:max_sentences])
    
    async def calculate_text_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate cosine similarity between two texts"""
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except ValueError:
            return 0.0
    
    async def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5
    ) -> List[int]:
        """Cluster texts using K-means"""
        if len(texts) < n_clusters:
            return list(range(len(texts)))
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            return cluster_labels.tolist()
        except ValueError:
            return [0] * len(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "initialized": self._initialized,
            "spacy_available": SPACY_AVAILABLE and self._spacy_nlp is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "loaded_pipelines": list(self._transformers_pipelines.keys()),
            "topic_models": len(self._topic_models),
            "privacy_mode": self._anonymize_entities
        }


# Global NLP processor instance
_nlp_processor: Optional[NLPProcessor] = None


async def get_nlp_processor() -> NLPProcessor:
    """Get the global NLP processor instance"""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor()
        await _nlp_processor.initialize()
    return _nlp_processor


async def analyze_text_batch(
    texts: List[str],
    **kwargs
) -> List[TextAnalysisResult]:
    """Convenience function for batch text analysis"""
    processor = await get_nlp_processor()
    return await processor.batch_analyze_texts(texts, **kwargs)


async def quick_sentiment_analysis(text: str) -> Tuple[float, SentimentLabel]:
    """Quick sentiment analysis without full text processing"""
    processor = await get_nlp_processor()
    return await processor._analyze_sentiment(text)


async def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
    """Quick keyword extraction"""
    processor = await get_nlp_processor()
    return processor._extract_keywords(text, max_keywords)