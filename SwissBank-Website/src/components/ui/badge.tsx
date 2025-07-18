import * as React from "react"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline"
}

function Badge({ className = "", variant = "default", children, ...props }: BadgeProps) {
  const baseClasses = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold"
  
  const variantClasses = {
    default: "border-transparent bg-yellow-400 text-black",
    secondary: "border-transparent bg-gray-600 text-white", 
    destructive: "border-transparent bg-red-600 text-white",
    outline: "border-gray-400 text-gray-300"
  }
  
  const combinedClasses = `${baseClasses} ${variantClasses[variant]} ${className}`.trim()
  
  return (
    <div className={combinedClasses} {...props}>
      {children}
    </div>
  )
}

export { Badge }

