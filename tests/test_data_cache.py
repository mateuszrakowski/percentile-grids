import os
import pickle
from unittest.mock import patch

from grids.utils.data_cache import (
    clear_cache,
    clear_model_cache,
    disk_cache,
    generate_cache_key,
)


class TestDataCache:
    """Test cases for data cache utilities."""

    def test_generate_cache_key(self):
        """Test cache key generation."""

        def test_function(a, b, c=10):
            return a + b + c

        # Test with different arguments
        key1 = generate_cache_key(test_function, (1, 2), {"c": 5})
        key2 = generate_cache_key(test_function, (1, 2), {"c": 10})
        key3 = generate_cache_key(test_function, (1, 2), {"c": 5})

        # Keys should be different for different arguments
        assert key1 != key2
        # Keys should be the same for same arguments
        assert key1 == key3
        # Keys should be valid MD5 hashes (32 hex characters)
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)

    def test_generate_cache_key_with_complex_objects(self):
        """Test cache key generation with complex objects."""

        def test_function(data):
            return sum(data)

        # Test with lists
        key1 = generate_cache_key(test_function, ([1, 2, 3],), {})
        key2 = generate_cache_key(test_function, ([1, 2, 3],), {})
        key3 = generate_cache_key(test_function, ([1, 2, 4],), {})

        assert key1 == key2
        assert key1 != key3

    def test_disk_cache_decorator(self, temp_dir):
        """Test disk cache decorator functionality."""
        cache_dir = os.path.join(temp_dir, "cache")

        @disk_cache(cache_dir)
        def test_function(x, y):
            return x + y

        # First call should compute and cache
        result1 = test_function(5, 3)
        assert result1 == 8

        # Check that cache file was created
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1
        assert cache_files[0].endswith(".pickle")

        # Second call should return cached result
        result2 = test_function(5, 3)
        assert result2 == 8

    def test_disk_cache_memory_cache(self, temp_dir):
        """Test that disk cache also uses memory cache."""
        cache_dir = os.path.join(temp_dir, "cache")

        call_count = 0

        @disk_cache(cache_dir)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use memory cache
        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

    def test_disk_cache_different_arguments(self, temp_dir):
        """Test that different arguments create different cache entries."""
        cache_dir = os.path.join(temp_dir, "cache")

        @disk_cache(cache_dir)
        def test_function(x, y):
            return x + y

        # Call with different arguments
        test_function(1, 2)
        test_function(3, 4)

        # Should create two cache files
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 2

    def test_disk_cache_pickle_error_handling(self, temp_dir):
        """Test handling of pickle errors in disk cache."""
        cache_dir = os.path.join(temp_dir, "cache")

        @disk_cache(cache_dir)
        def test_function(x):
            return x * 2

        # Mock pickle to raise an error
        with patch("pickle.dump") as mock_dump:
            mock_dump.side_effect = pickle.PickleError("Test error")

            # Function should still work despite pickle error
            result = test_function(5)
            assert result == 10

    def test_clear_cache(self, temp_dir):
        """Test clearing cache directory."""
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Create some test files
        test_files = ["file1.pickle", "file2.pickle", "file3.txt"]
        for file_name in test_files:
            with open(os.path.join(cache_dir, file_name), "w") as f:
                f.write("test content")

        # Clear cache
        clear_cache(cache_dir)

        # Check that all files were removed
        assert len(os.listdir(cache_dir)) == 0

    def test_clear_model_cache(self, temp_dir):
        """Test clearing model cache directory."""
        model_cache_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_cache_dir, exist_ok=True)

        # Create some test model files
        test_files = ["model1.rds", "model2.rds", "info1.json"]
        for file_name in test_files:
            with open(os.path.join(model_cache_dir, file_name), "w") as f:
                f.write("test content")

        # Clear model cache
        clear_model_cache(model_cache_dir)

        # Check that all files were removed
        assert len(os.listdir(model_cache_dir)) == 0

    def test_clear_cache_nonexistent_directory(self, temp_dir):
        """Test clearing cache from non-existent directory."""
        cache_dir = os.path.join(temp_dir, "nonexistent")

        # Should not raise an error
        clear_cache(cache_dir)

    def test_clear_model_cache_nonexistent_directory(self, temp_dir):
        """Test clearing model cache from non-existent directory."""
        model_cache_dir = os.path.join(temp_dir, "nonexistent")

        # Should not raise an error
        clear_model_cache(model_cache_dir)

    def test_disk_cache_with_kwargs(self, temp_dir):
        """Test disk cache with keyword arguments."""
        cache_dir = os.path.join(temp_dir, "cache")

        @disk_cache(cache_dir)
        def test_function(x, y=10, z=20):
            return x + y + z

        # Test with different keyword argument combinations
        result1 = test_function(5, y=15, z=25)
        result2 = test_function(5, z=25, y=15)  # Same args, different order
        result3 = test_function(5, y=20, z=30)  # Different args

        assert result1 == 45
        assert result1 == result2  # Should be cached
        assert result3 == 55  # Different result

    def test_disk_cache_function_source_code_changes(self):
        """Test that cache keys change when function source code changes."""

        # Define function with different implementations
        def test_function_v1(x):
            return x * 2

        def test_function_v2(x):
            return x * 3  # Different implementation

        # Generate keys for same arguments
        key1 = generate_cache_key(test_function_v1, (5,), {})
        key2 = generate_cache_key(test_function_v2, (5,), {})

        # Keys should be different due to different source code
        assert key1 != key2
