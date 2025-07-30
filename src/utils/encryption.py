"""
Comprehensive Encryption Utilities

This module provides enterprise-grade encryption capabilities:
- AES-256-GCM encryption for data at rest
- Secure key derivation using PBKDF2 and Argon2
- File-based encrypted storage with integrity verification
- Salt generation and management
- Secure random number generation
- Key rotation and management utilities
"""

import os
import secrets
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, HashingError

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"


class KeyDerivationFunction(str, Enum):
    """Supported key derivation functions"""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data"""
    algorithm: EncryptionAlgorithm
    kdf: KeyDerivationFunction
    salt: bytes
    nonce: bytes
    created_at: datetime
    key_version: int
    checksum: str


class SecureRandom:
    """Cryptographically secure random number generator"""
    
    @staticmethod
    def generate_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_hex(length: int) -> str:
        """Generate cryptographically secure random hex string"""
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_urlsafe(length: int) -> str:
        """Generate cryptographically secure URL-safe string"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Generate cryptographic salt"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_nonce_gcm() -> bytes:
        """Generate nonce for AES-GCM (12 bytes recommended)"""
        return secrets.token_bytes(12)
    
    @staticmethod
    def generate_nonce_chacha20() -> bytes:
        """Generate nonce for ChaCha20-Poly1305 (12 bytes)"""
        return secrets.token_bytes(12)


class KeyDerivation:
    """Key derivation utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self._argon2_hasher = PasswordHasher(
            time_cost=self.settings.encryption.argon2_time_cost,
            memory_cost=self.settings.encryption.argon2_memory_cost,
            parallelism=self.settings.encryption.argon2_parallelism,
        )
    
    def derive_key_pbkdf2(
        self,
        password: Union[str, bytes],
        salt: bytes,
        iterations: Optional[int] = None,
        key_length: int = 32
    ) -> bytes:
        """Derive key using PBKDF2-HMAC-SHA256"""
        
        if isinstance(password, str):
            password = password.encode()
        
        if iterations is None:
            iterations = self.settings.encryption.pbkdf2_iterations
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        return kdf.derive(password)
    
    def derive_key_scrypt(
        self,
        password: Union[str, bytes],
        salt: bytes,
        n: int = 2**14,  # CPU/memory cost parameter
        r: int = 8,      # Block size parameter
        p: int = 1,      # Parallelization parameter
        key_length: int = 32
    ) -> bytes:
        """Derive key using Scrypt"""
        
        if isinstance(password, str):
            password = password.encode()
        
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            n=n,
            r=r,
            p=p,
            backend=default_backend()
        )
        
        return kdf.derive(password)
    
    def hash_password_argon2(self, password: str) -> str:
        """Hash password using Argon2id"""
        try:
            return self._argon2_hasher.hash(password)
        except HashingError as e:
            logger.error("Password hashing failed", error=str(e))
            raise
    
    def verify_password_argon2(self, hashed_password: str, password: str) -> bool:
        """Verify password against Argon2 hash"""
        try:
            self._argon2_hasher.verify(hashed_password, password)
            return True
        except VerifyMismatchError:
            return False
        except Exception as e:
            logger.error("Password verification failed", error=str(e))
            return False


class SymmetricEncryption:
    """Symmetric encryption using AES-256-GCM and ChaCha20-Poly1305"""
    
    def __init__(self):
        self.key_derivation = KeyDerivation()
    
    def encrypt_aes_gcm(
        self,
        data: Union[str, bytes],
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-256-GCM
        Returns: (encrypted_data, nonce, tag)
        """
        
        if isinstance(data, str):
            data = data.encode()
        
        if len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")
        
        nonce = SecureRandom.generate_nonce_gcm()
        aesgcm = AESGCM(key)
        
        encrypted_data = aesgcm.encrypt(nonce, data, associated_data)
        
        return encrypted_data, nonce, b""  # GCM includes authentication tag in encrypted_data
    
    def decrypt_aes_gcm(
        self,
        encrypted_data: bytes,
        key: bytes,
        nonce: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """Decrypt data using AES-256-GCM"""
        
        if len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")
        
        aesgcm = AESGCM(key)
        
        try:
            decrypted_data = aesgcm.decrypt(nonce, encrypted_data, associated_data)
            return decrypted_data
        except Exception as e:
            logger.error("AES-GCM decryption failed", error=str(e))
            raise ValueError("Decryption failed - data may be corrupted or key is incorrect")
    
    def encrypt_chacha20(
        self,
        data: Union[str, bytes],
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt data using ChaCha20-Poly1305
        Returns: (encrypted_data, nonce)
        """
        
        if isinstance(data, str):
            data = data.encode()
        
        if len(key) != 32:
            raise ValueError("ChaCha20 requires 32-byte key")
        
        nonce = SecureRandom.generate_nonce_chacha20()
        chacha = ChaCha20Poly1305(key)
        
        encrypted_data = chacha.encrypt(nonce, data, associated_data)
        
        return encrypted_data, nonce
    
    def decrypt_chacha20(
        self,
        encrypted_data: bytes,
        key: bytes,
        nonce: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        
        if len(key) != 32:
            raise ValueError("ChaCha20 requires 32-byte key")
        
        chacha = ChaCha20Poly1305(key)
        
        try:
            decrypted_data = chacha.decrypt(nonce, encrypted_data, associated_data)
            return decrypted_data
        except Exception as e:
            logger.error("ChaCha20-Poly1305 decryption failed", error=str(e))
            raise ValueError("Decryption failed - data may be corrupted or key is incorrect")


class AsymmetricEncryption:
    """Asymmetric encryption using RSA"""
    
    @staticmethod
    def generate_rsa_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair
        Returns: (private_key_pem, public_key_pem)
        """
        
        if key_size < 2048:
            raise ValueError("RSA key size must be at least 2048 bits")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    @staticmethod
    def encrypt_rsa(data: Union[str, bytes], public_key_pem: bytes) -> bytes:
        """Encrypt data using RSA-OAEP"""
        
        if isinstance(data, str):
            data = data.encode()
        
        public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
        
        # RSA can only encrypt data smaller than key size - padding
        # For larger data, use hybrid encryption (RSA + AES)
        max_chunk_size = (public_key.key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2
        
        if len(data) > max_chunk_size:
            raise ValueError(f"Data too large for RSA encryption. Max size: {max_chunk_size} bytes")
        
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_data
    
    @staticmethod
    def decrypt_rsa(encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data using RSA-OAEP"""
        
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
        
        try:
            decrypted_data = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data
        except Exception as e:
            logger.error("RSA decryption failed", error=str(e))
            raise ValueError("RSA decryption failed - data may be corrupted or key is incorrect")


class SecureStorage:
    """Secure file storage with encryption and integrity verification"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.settings = get_settings()
        self.symmetric_encryption = SymmetricEncryption()
        self.key_derivation = KeyDerivation()
        
        if master_key:
            self._master_key = master_key
        else:
            # Derive master key from settings
            master_key_str = self.settings.encryption.master_key.get_secret_value()
            salt = self.settings.encryption.key_salt.get_secret_value().encode()
            self._master_key = self.key_derivation.derive_key_pbkdf2(master_key_str, salt)
        
        self._key_version = 1  # For key rotation support
    
    def _generate_file_key(self, filename: str, salt: bytes) -> bytes:
        """Generate file-specific encryption key"""
        # Combine master key with filename and salt for file-specific key
        context = f"{filename}:{salt.hex()}".encode()
        return self.key_derivation.derive_key_pbkdf2(self._master_key + context, salt)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data"""
        return hashlib.sha256(data).hexdigest()
    
    def encrypt_file(
        self,
        file_path: Union[str, Path],
        data: Union[str, bytes, Dict[str, Any]],
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    ) -> bool:
        """Encrypt and save data to file"""
        
        file_path = Path(file_path)
        
        # Convert data to bytes if needed
        if isinstance(data, dict):
            data_bytes = json.dumps(data, ensure_ascii=False).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = data
        
        # Generate salt and derive file-specific key
        salt = SecureRandom.generate_salt()
        file_key = self._generate_file_key(str(file_path), salt)
        
        try:
            # Encrypt data
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data, nonce, _ = self.symmetric_encryption.encrypt_aes_gcm(data_bytes, file_key)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                encrypted_data, nonce = self.symmetric_encryption.encrypt_chacha20(data_bytes, file_key)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
            
            # Create metadata
            metadata = EncryptionMetadata(
                algorithm=algorithm,
                kdf=KeyDerivationFunction.PBKDF2,
                salt=salt,
                nonce=nonce,
                created_at=datetime.now(timezone.utc),
                key_version=self._key_version,
                checksum=self._calculate_checksum(encrypted_data)
            )
            
            # Prepare file structure
            file_data = {
                "metadata": {
                    "algorithm": metadata.algorithm.value,
                    "kdf": metadata.kdf.value,
                    "salt": metadata.salt.hex(),
                    "nonce": metadata.nonce.hex(),
                    "created_at": metadata.created_at.isoformat(),
                    "key_version": metadata.key_version,
                    "checksum": metadata.checksum,
                },
                "encrypted_data": encrypted_data.hex(),
            }
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first, then rename for atomic operation
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            
            with open(temp_path, 'w') as f:
                json.dump(file_data, f, indent=2)
            
            # Set restrictive permissions
            temp_path.chmod(0o600)
            
            # Atomic rename
            temp_path.rename(file_path)
            
            logger.info(f"File encrypted and saved", file_path=str(file_path), algorithm=algorithm.value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt file", file_path=str(file_path), error=str(e))
            return False
    
    def decrypt_file(
        self,
        file_path: Union[str, Path],
        return_dict: bool = False
    ) -> Optional[Union[bytes, str, Dict[str, Any]]]:
        """Decrypt and load data from file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Encrypted file not found", file_path=str(file_path))
            return None
        
        try:
            # Load file data
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # Extract metadata
            metadata_dict = file_data["metadata"]
            metadata = EncryptionMetadata(
                algorithm=EncryptionAlgorithm(metadata_dict["algorithm"]),
                kdf=KeyDerivationFunction(metadata_dict["kdf"]),
                salt=bytes.fromhex(metadata_dict["salt"]),
                nonce=bytes.fromhex(metadata_dict["nonce"]),
                created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                key_version=metadata_dict["key_version"],
                checksum=metadata_dict["checksum"]
            )
            
            # Extract encrypted data
            encrypted_data = bytes.fromhex(file_data["encrypted_data"])
            
            # Verify checksum
            calculated_checksum = self._calculate_checksum(encrypted_data)
            if calculated_checksum != metadata.checksum:
                raise ValueError("File integrity check failed - data may be corrupted")
            
            # Derive file key
            file_key = self._generate_file_key(str(file_path), metadata.salt)
            
            # Decrypt data
            if metadata.algorithm == EncryptionAlgorithm.AES_256_GCM:
                decrypted_data = self.symmetric_encryption.decrypt_aes_gcm(encrypted_data, file_key, metadata.nonce)
            elif metadata.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                decrypted_data = self.symmetric_encryption.decrypt_chacha20(encrypted_data, file_key, metadata.nonce)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {metadata.algorithm}")
            
            # Convert back to original format
            if return_dict:
                try:
                    return json.loads(decrypted_data.decode())
                except json.JSONDecodeError:
                    # If not valid JSON, return as string
                    return decrypted_data.decode()
            else:
                return decrypted_data
            
        except Exception as e:
            logger.error(f"Failed to decrypt file", file_path=str(file_path), error=str(e))
            return None
    
    def rotate_file_key(self, file_path: Union[str, Path]) -> bool:
        """Rotate encryption key for a file by re-encrypting with new key"""
        
        # Decrypt with old key
        data = self.decrypt_file(file_path)
        
        if data is None:
            logger.error(f"Cannot rotate key - failed to decrypt file", file_path=str(file_path))
            return False
        
        # Update key version
        old_key_version = self._key_version
        self._key_version += 1
        
        # Re-encrypt with new key version
        success = self.encrypt_file(file_path, data)
        
        if success:
            logger.info(f"File key rotated", file_path=str(file_path), old_version=old_key_version, new_version=self._key_version)
        else:
            # Restore old key version on failure
            self._key_version = old_key_version
        
        return success
    
    def verify_file_integrity(self, file_path: Union[str, Path]) -> bool:
        """Verify file integrity without decrypting"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            encrypted_data = bytes.fromhex(file_data["encrypted_data"])
            stored_checksum = file_data["metadata"]["checksum"]
            
            calculated_checksum = self._calculate_checksum(encrypted_data)
            
            return calculated_checksum == stored_checksum
            
        except Exception as e:
            logger.error(f"Failed to verify file integrity", file_path=str(file_path), error=str(e))
            return False


class EncryptionManager:
    """High-level encryption manager for the application"""
    
    def __init__(self):
        self.settings = get_settings()
        self.secure_storage = SecureStorage()
        self.symmetric_encryption = SymmetricEncryption()
        self.asymmetric_encryption = AsymmetricEncryption()
        self.key_derivation = KeyDerivation()
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]], context: str = "") -> Optional[str]:
        """Encrypt sensitive data and return base64-encoded result"""
        
        try:
            # Convert to JSON if dict
            if isinstance(data, dict):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)
            
            # Generate context-specific key
            salt = SecureRandom.generate_salt()
            master_key = self.settings.encryption.master_key.get_secret_value()
            context_key = self.key_derivation.derive_key_pbkdf2(f"{master_key}:{context}", salt)
            
            # Encrypt data
            encrypted_data, nonce, _ = self.symmetric_encryption.encrypt_aes_gcm(data_str, context_key)
            
            # Combine salt, nonce, and encrypted data
            combined_data = salt + nonce + encrypted_data
            
            # Return hex-encoded result
            return combined_data.hex()
            
        except Exception as e:
            logger.error(f"Failed to encrypt sensitive data", context=context, error=str(e))
            return None
    
    def decrypt_sensitive_data(self, encrypted_hex: str, context: str = "") -> Optional[Union[str, Dict[str, Any]]]:
        """Decrypt sensitive data from hex-encoded string"""
        
        try:
            # Decode from hex
            combined_data = bytes.fromhex(encrypted_hex)
            
            # Extract components
            salt = combined_data[:32]  # First 32 bytes
            nonce = combined_data[32:44]  # Next 12 bytes
            encrypted_data = combined_data[44:]  # Remaining bytes
            
            # Derive context-specific key
            master_key = self.settings.encryption.master_key.get_secret_value()
            context_key = self.key_derivation.derive_key_pbkdf2(f"{master_key}:{context}", salt)
            
            # Decrypt data
            decrypted_data = self.symmetric_encryption.decrypt_aes_gcm(encrypted_data, context_key, nonce)
            
            # Try to parse as JSON, otherwise return as string
            try:
                return json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Failed to decrypt sensitive data", context=context, error=str(e))
            return None
    
    def secure_delete_file(self, file_path: Union[str, Path]) -> bool:
        """Securely delete a file by overwriting with random data"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return True
        
        try:
            # Get file size
            file_size = file_path.stat().st_size
            
            # Overwrite with random data multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(SecureRandom.generate_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            file_path.unlink()
            
            logger.info(f"File securely deleted", file_path=str(file_path))
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete file", file_path=str(file_path), error=str(e))
            return False


# Global encryption manager instance
encryption_manager: Optional[EncryptionManager] = None


def get_encryption_manager() -> EncryptionManager:
    """Get encryption manager singleton"""
    global encryption_manager
    if encryption_manager is None:
        encryption_manager = EncryptionManager()
    return encryption_manager