import os
import fnmatch
import re

def should_skip_file(filename):
    # Root level files to skip
    root_level_skips = {
        'LICENSE',
        'LICENSE.txt',
        'LICENSE.md',
    }
    
    # If it's a root level file that should be skipped
    if os.path.dirname(filename) == '.' and os.path.basename(filename) in root_level_skips:
        return True

    # Skip patterns for files we don't want to include
    skip_patterns = [
        '*.min.js',      # Minified JavaScript
        '*.pyc',         # Python compiled files
        '*.pyo',         # Python optimized files
        '*.pyd',         # Python DLL files
        '*.so',          # Shared libraries
        '*.dll',         # DLL files
        '*.dylib',       # Dynamic libraries
        '*.class',       # Java compiled files
        '*.exe',         # Executables
        '*.o',           # Object files
        '*.a',           # Static libraries
        '*.lib',         # Library files
        '*.zip',         # Archives
        '*.tar',         # Archives
        '*.gz',          # Compressed files
        '*.rar',         # Compressed files
        '*.7z',          # Compressed files
        '*.map',         # Source map files
        'README*',       # README files
        'CHANGELOG*',    # Changelog files
        'requirements*.txt',  # Requirements files
        '*.svg',         # SVG files
        '*.png',         # Images
        '*.jpg',         # Images
        '*.jpeg',        # Images
        '*.gif',         # Images
        '*.ico',         # Icons
        '*.woff',        # Fonts
        '*.woff2',       # Fonts
        '*.ttf',         # Fonts
        '*.eot',         # Fonts
        '*.css',         # CSS files
        '*.scss',        # SCSS files
        '*.sass',        # SASS files
        '*.less',        # LESS files
        '*.json',        # JSON files
        'package-lock.json',  # NPM lock file
        'yarn.lock',     # Yarn lock file
        'pnpm-lock.yaml', # PNPM lock file
    ]
    
    # Skip files that look like build artifacts (containing hashes)
    if re.search(r'-[a-zA-Z0-9]{8,}\.', filename):
        return True
        
    return any(fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in skip_patterns)

def should_skip_directory(dirpath):
    # Directories to skip
    skip_dirs = {
        'venv',
        'env',
        'node_modules',
        '__pycache__',
        '.git',
        '.idea',
        '.vscode',
        'site-packages',
        'dist',
        'build',
        'tests',
        'test',
        'docs',
        'examples',
        'assets',      # Web assets directory
        'static',      # Static files directory
        'public',      # Public assets
        'images',      # Image directories
        'img',         # Image directories
        'fonts',       # Font directories
        'css',         # CSS directories
        'scss',        # SCSS directories
        'styles',      # Style directories
    }
    
    dir_name = os.path.basename(dirpath)
    return dir_name.lower() in skip_dirs

def collect_code(start_path='.'):
    with open('codebase.txt', 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(start_path):
            # Remove directories we want to skip
            dirs[:] = [d for d in dirs if not should_skip_directory(os.path.join(root, d)) and not d.startswith('.')]
            
            for file in sorted(files):
                if file.startswith('.'):
                    continue
                    
                filepath = os.path.join(root, file)
                
                if should_skip_file(filepath):
                    continue
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                    # Write file path with clear demarcation
                    outfile.write('\n' + '=' * 80 + '\n')
                    outfile.write(f'FILE: {filepath}\n')
                    outfile.write('=' * 80 + '\n\n')
                    outfile.write(content)
                    outfile.write('\n')
                except (UnicodeDecodeError, IOError):
                    # Skip binary files or files that can't be read as text
                    continue

if __name__ == '__main__':
    collect_code() 