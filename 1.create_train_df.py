from pathlib import Path
import argparse
import pandas as pd
import cv2


def build_dataframe(root_path: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    rows = []
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            parent = p.parent
            label = parent.name
            split = parent.parent.name if parent.parent is not None else ''
            img = cv2.imread(str(p))
            if img is None:
                height = None
                width = None
                channel_type = 'unknown'
            else:
                height, width = img.shape[:2]
                if len(img.shape) == 2:
                    channel_type = 'grayscale'
                elif len(img.shape) == 3:
                    if img.shape[2] == 3:
                        channel_type = 'rgb'
                    elif img.shape[2] == 4:
                        channel_type = 'rgba'
                    else:
                        channel_type = f'{img.shape[2]} channels'
                else:
                    channel_type = 'unknown'
            # Dosya boyutu (byte)
            file_size = p.stat().st_size
            # Sınıflandırma
            if file_size < 50*1024:
                size_class = '0-50kb'
            elif file_size < 150*1024:
                size_class = '50-150kb'
            elif file_size < 500*1024:
                size_class = '150-500kb'
            else:
                size_class = '500kb+'
            rows.append({
                'filepath': str(p.resolve()),
                'filename': p.name,
                'label': label,
                'split': split,
                'width': width,
                'height': height,
                'size_class': size_class,
                'channel_type': channel_type,
            })

    df = pd.DataFrame(rows)
    return df


def main():
    default_root = r"C:\PythonProject\2.donem projeleri\SayisalGoruntu\SkinCancerISIC"
    parser = argparse.ArgumentParser(description='Create train_df from image dataset folder')
    parser.add_argument('--root', '-r', default=default_root, help='Root folder to scan')
    parser.add_argument('--out', '-o', default=None, help='Output CSV path (default: <root>/train_df.csv)')
    args = parser.parse_args()

    root = Path(args.root)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = root / 'train_df.csv'

    print(f"Scanning root: {root}")
    df = build_dataframe(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    #print(f"Total images found: {len(df)}")
    print(f"Saved CSV to: {out_path}")


if __name__ == '__main__':
    main()
    # Sonuçları göster
    csv_path = r"C:\PythonProject\2.donem projeleri\SayisalGoruntu\SkinCancerISIC\train_df.csv"
    try:
        df = pd.read_csv(csv_path)
        print("\nİlk 10 kayıt:")
        print(df.head(10))
        print(f"\nToplam kayıt sayısı: {len(df)}")
    except Exception as e:
        print(f"CSV okunamadı: {e}")
