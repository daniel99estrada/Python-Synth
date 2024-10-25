// src/components/ui/button.js
const Button = ({ children, onClick, className }) => (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded focus:outline-none ${className}`}
    >
      {children}
    </button>
  );
  
  export default Button;
  