import { cn } from "@/lib/utils";

interface RadiologyImageCardProps {
  src: string;
  alt: string;
  label?: string;
  caption?: string;
  className?: string;
  imageClassName?: string;
  overlayClassName?: string;
  scanLine?: boolean;
}

export const RadiologyImageCard = ({
  src,
  alt,
  label,
  caption,
  className,
  imageClassName,
  overlayClassName,
  scanLine = true,
}: RadiologyImageCardProps) => (
  <figure className={cn("radiology-image-card group", className)}>
    <img src={src} alt={alt} loading="lazy" className={cn("radiology-image", imageClassName)} />
    <span className="radiology-image-vignette" aria-hidden="true" />
    {scanLine && <span className="radiology-scan-line" aria-hidden="true" />}
    {(label || caption) && (
      <figcaption className={cn("radiology-image-caption", overlayClassName)}>
        {label && <span className="radiology-image-label">{label}</span>}
        {caption && <span className="radiology-image-copy">{caption}</span>}
      </figcaption>
    )}
  </figure>
);

