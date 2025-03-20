
async def identify_user(self, audio_file_path, user_text=None):
        """Identify a user based on their voice and optionally their text"""
        
        if not self.user_profiles:
            return None, 0.0, "No profiles in database"  # No profiles to match against
        
        # Set up audio config for recognition
        audio_config = AudioConfig(filename=audio_file_path)
        
        # Create speaker recognizer
        recognizer = SpeakerRecognizer(self.speech_config, audio_config)
        
        # Get all voice profile IDs
        voice_profile_ids = []
        id_mapping = {}  # Map Azure profile IDs to our internal user IDs
        
        for user_id, profile in self.user_profiles.items():
            voice_profile_id = profile.get("voice_profile_id")
            if voice_profile_id:
                voice_profile_ids.append(voice_profile_id)
                id_mapping[voice_profile_id] = user_id
        
        if not voice_profile_ids:
            return None, 0.0, "No valid voice profiles found"
            
        # Perform voice identification against all profiles
        voice_result = await recognizer.recognize_once_async(voice_profile_ids)
        
        if not voice_result or not voice_result.profile_id:
            return None, 0.0, "Voice not recognized"
            
        # Get our internal user ID from the matched Azure profile ID
        user_id = id_mapping.get(voice_result.profile_id)
        if not user_id:
            return None, 0.0, "Profile ID mapping error"
            
        voice_confidence = voice_result.score
        text_similarity = 0.0
        combined_score = voice_confidence
        
        # Get the user profile
        profile = self.user_profiles[user_id]
        
        # If we have text, enhance the match with text similarity
        if user_text and "text_embedding" in profile:
            text_embedding = self._generate_text_embedding(user_text)
            text_similarity = self._calculate_text_similarity(
                text_embedding, 
                profile["text_embedding"]
            )
            
            # Update user text samples
            if "user_text_samples" not in profile:
                profile["user_text_samples"] = []
            profile["user_text_samples"].append(user_text)
            
            # Update text embedding
            profile["text_embedding"] = self._update_text_embedding(
                profile["text_embedding"], 
                text_embedding
            )
            
            # Calculate combined confidence score
            combined_score = 0.7 * voice_confidence + 0.3 * text_similarity
        
        # Update user profile data
        profile["last_seen"] = datetime.now().isoformat()
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1
        self._save_profiles()
        
        return user_id, combined_score, "User recognized"