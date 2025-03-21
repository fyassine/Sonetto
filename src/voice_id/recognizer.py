import os
import json
import uuid
from datetime import datetime
import numpy as np
import hashlib

class VoiceIdentifier:
    """Class to handle voice identification and profile management."""
    
    def __init__(self, speech_key, speech_region):
        """Initialize the voice identifier."""
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.user_profiles = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load user profiles from disk."""
        profiles_path = os.path.join(os.path.dirname(__file__), "profiles.json")
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'r') as f:
                    self.user_profiles = json.load(f)
                print(f"Loaded {len(self.user_profiles)} user profiles")
            except Exception as e:
                print(f"Error loading profiles: {str(e)}")
                self.user_profiles = {}
    
    def _save_profiles(self):
        """Save user profiles to disk."""
        profiles_path = os.path.join(os.path.dirname(__file__), "profiles.json")
        try:
            with open(profiles_path, 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
            print(f"Saved {len(self.user_profiles)} user profiles")
        except Exception as e:
            print(f"Error saving profiles: {str(e)}")
    
    def _generate_voice_embedding(self, audio_file_path):
        """
        Generate a simple voice embedding from an audio file.
        
        This is a simplified implementation that uses file characteristics
        as a placeholder for actual voice embeddings.
        """
        try:
            # In a real implementation, this would use a proper voice embedding model
            # For now, we'll use a hash of the file as a simple placeholder
            with open(audio_file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            # Convert hash to a list of numbers (our "embedding")
            embedding = []
            for i in range(0, len(file_hash), 2):
                if i+1 < len(file_hash):
                    embedding.append(int(file_hash[i:i+2], 16) / 255.0)
            
            return embedding
        except Exception as e:
            print(f"Error generating voice embedding: {str(e)}")
            return [0.0] * 16  # Return a zero embedding
    
    def _generate_text_embedding(self, text):
        """
        Generate a simple text embedding.
        
        This is a simplified implementation that uses character frequencies
        as a placeholder for actual text embeddings.
        """
        if not text:
            return [0.0] * 26
            
        # Count character frequencies (a-z)
        char_counts = [0] * 26
        text = text.lower()
        
        for char in text:
            if 'a' <= char <= 'z':
                index = ord(char) - ord('a')
                char_counts[index] += 1
        
        # Normalize
        total = sum(char_counts) or 1
        return [count / total for count in char_counts]
    
    def _calculate_voice_similarity(self, embedding1, embedding2):
        """Calculate similarity between two voice embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_text_similarity(self, embedding1, embedding2):
        """Calculate similarity between two text embeddings."""
        return self._calculate_voice_similarity(embedding1, embedding2)
    
    def _update_voice_embedding(self, old_embedding, new_embedding):
        """Update a voice embedding with new data."""
        if not old_embedding:
            return new_embedding
            
        # Convert to numpy arrays
        old_vec = np.array(old_embedding)
        new_vec = np.array(new_embedding)
        
        # Calculate updated embedding (70% old, 30% new)
        updated_vec = 0.7 * old_vec + 0.3 * new_vec
        
        return updated_vec.tolist()
    
    def _update_text_embedding(self, old_embedding, new_embedding):
        """Update a text embedding with new data."""
        return self._update_voice_embedding(old_embedding, new_embedding)
    
    async def identify_user(self, audio_file_path, user_text=None):
        """Identify a user based on their voice and optionally their text"""
        
        if not self.user_profiles:
            return None, 0.0, "No profiles in database"  # No profiles to match against
        
        # Generate voice embedding for the input audio
        voice_embedding = self._generate_voice_embedding(audio_file_path)
        
        # Find the best matching profile
        best_match_id = None
        best_match_score = 0.0
        
        for user_id, profile in self.user_profiles.items():
            if "voice_embedding" not in profile:
                continue
                
            # Calculate voice similarity
            voice_similarity = self._calculate_voice_similarity(
                voice_embedding, 
                profile["voice_embedding"]
            )
            
            combined_score = voice_similarity
            
            # If we have text, enhance the match with text similarity
            if user_text and "text_embedding" in profile:
                text_embedding = self._generate_text_embedding(user_text)
                text_similarity = self._calculate_text_similarity(
                    text_embedding, 
                    profile["text_embedding"]
                )
                
                # Calculate combined score (70% voice, 30% text)
                combined_score = 0.7 * voice_similarity + 0.3 * text_similarity
            
            # Check if this is the best match so far
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_id = user_id
        
        # If we found a match, update the profile
        if best_match_id:
            profile = self.user_profiles[best_match_id]
            
            # Update user profile data
            profile["last_seen"] = datetime.now().isoformat()
            profile["interaction_count"] = profile.get("interaction_count", 0) + 1
            
            # Update voice embedding
            profile["voice_embedding"] = self._update_voice_embedding(
                profile["voice_embedding"], 
                voice_embedding
            )
            
            # Update text embedding if available
            if user_text:
                if "user_text_samples" not in profile:
                    profile["user_text_samples"] = []
                profile["user_text_samples"].append(user_text)
                
                text_embedding = self._generate_text_embedding(user_text)
                if "text_embedding" in profile:
                    profile["text_embedding"] = self._update_text_embedding(
                        profile["text_embedding"], 
                        text_embedding
                    )
                else:
                    profile["text_embedding"] = text_embedding
            
            self._save_profiles()
            
            return best_match_id, best_match_score, "User recognized"
        
        return None, 0.0, "No matching user found"
    
    async def create_voice_profile(self, user_id, audio_file_path, user_text=None):
        """Create a new voice profile for a user."""
        try:
            # Generate voice embedding
            voice_embedding = self._generate_voice_embedding(audio_file_path)
            
            # Create a new user profile
            new_profile = {
                "voice_embedding": voice_embedding,
                "created_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "interaction_count": 1,
                "voice_samples": [audio_file_path]
            }
            
            # Add text data if available
            if user_text:
                new_profile["user_text_samples"] = [user_text]
                new_profile["text_embedding"] = self._generate_text_embedding(user_text)
            
            # Save the profile
            self.user_profiles[user_id] = new_profile
            self._save_profiles()
            
            print(f"Created new voice profile for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error creating voice profile: {str(e)}")
            return False
    
    def update_voice_profile(self, user_id, audio_file_path):
        """Update an existing voice profile with new audio."""
        try:
            if user_id not in self.user_profiles:
                print(f"User {user_id} not found")
                return False
                
            profile = self.user_profiles[user_id]
            
            # Generate voice embedding
            voice_embedding = self._generate_voice_embedding(audio_file_path)
            
            # Update the voice embedding
            if "voice_embedding" in profile:
                profile["voice_embedding"] = self._update_voice_embedding(
                    profile["voice_embedding"], 
                    voice_embedding
                )
            else:
                profile["voice_embedding"] = voice_embedding
            
            # Add to voice samples
            if "voice_samples" not in profile:
                profile["voice_samples"] = []
            profile["voice_samples"].append(audio_file_path)
            
            # Update last seen
            profile["last_seen"] = datetime.now().isoformat()
            
            # Save profiles
            self._save_profiles()
            
            print(f"Updated voice profile for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error updating voice profile: {str(e)}")
            return False
    
    def get_user_data(self, user_id):
        """Get user data for a specific user ID."""
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found")
            return {}
            
        return self.user_profiles[user_id]

# Function to create a global instance of the VoiceIdentifier
def create_voice_identifier(speech_key, speech_region):
    """Create and return a VoiceIdentifier instance."""
    return VoiceIdentifier(speech_key, speech_region)

# Simplified function for direct identification (without class instance)
async def identify_user(audio_file_path, user_text=None):
    """Identify a user based on their voice (simplified function)."""
    try:
        from relay import voice_identifier
        
        # If voice_identifier is not initialized yet, return no match
        if not voice_identifier:
            return None, 0.0, "Voice identifier not initialized"
        
        # Call the instance method
        return await voice_identifier.identify_user(audio_file_path, user_text)
    except (ImportError, AttributeError) as e:
        print(f"Error in identify_user: {str(e)}")
        return None, 0.0, f"Error: {str(e)}"