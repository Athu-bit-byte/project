// Extract React and ReactDOM
const { useState, useEffect } = React;

// --- History App Component ---
function HistoryApp({ userId }) {
    const [historyData, setHistoryData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await fetch(`/api/history/${userId}`);
                if (!response.ok) throw new Error('Failed to fetch history');
                const data = await response.json();
                setHistoryData(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (userId) {
            fetchHistory();
        }
    }, [userId]);

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '3rem' }}>
                <i className="fas fa-spinner fa-spin" style={{ fontSize: '3rem', color: 'var(--primary-green)' }}></i>
                <p>Loading your history...</p>
            </div>
        );
    }

    if (error) {
        return <p className="error">Error loading history: {error}</p>;
    }

    const diseaseHistory = historyData.filter(item => item.type === 'disease');
    const cropHistory = historyData.filter(item => item.type === 'crop');

    return (
        <div>
            {/* Disease History Section */}
            <section className="history-section">
                <h2>Disease Detection History</h2>
                {diseaseHistory.length === 0 ? (
                    <div className="empty-state"><p>No disease detection records yet.</p></div>
                ) : (
                    <div className="history-grid">
                        {diseaseHistory.map((item, index) => (
                            <div key={index} className="history-card">
                                <div className="history-date">{new Date(item.date).toLocaleDateString()}</div>
                                <div className="history-detail"><strong>Crop:</strong> {item.crop}</div>
                                <div className="history-detail"><strong>Result:</strong> {item.result}</div>
                            </div>
                        ))}
                    </div>
                )}
            </section>

            {/* Crop History Section */}
            <section className="history-section">
                <h2>Crop Recommendation History</h2>
                {cropHistory.length === 0 ? (
                    <div className="empty-state"><p>No crop recommendation records yet.</p></div>
                ) : (
                    <div className="history-grid">
                        {cropHistory.map((item, index) => {
                            let crops = [];
                            try { crops = JSON.parse(item.result); } catch (e) { crops = [item.result]; }
                            return (
                                <div key={index} className="history-card">
                                    <div className="history-date">{new Date(item.date).toLocaleDateString()}</div>
                                    <div className="history-detail"><strong>Top Choice:</strong> {item.crop}</div>
                                    <div className="history-detail"><strong>Full Recommendations:</strong> {crops.join(', ')}</div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </section>
        </div>
    );
}

// Ensure the standard checkAuthStatus logic injects the React HistoryApp
window.renderReactHistory = async function () {
    const historyContainer = document.getElementById('react-history-root');
    if (!historyContainer) return; // Not on the history page

    const user = await checkAuthStatus();
    if (!user) {
        window.location.href = '/login?redirect=history';
        return;
    }

    const root = ReactDOM.createRoot(historyContainer);
    root.render(<HistoryApp userId={user.id} />);
};

// --- Government Schemes App Component ---
function SchemesApp() {
    return (
        <div>
            <h2>Government Scheme Information</h2>
            <p>Browse through government schemes available for farmers. (Content coming soon)</p>

            <div className="empty-state">
                <div className="empty-state-icon">
                    <i className="fas fa-seedling"></i>
                </div>
                <h3>Information Coming Soon</h3>
                <p>We are currently gathering comprehensive information on government agricultural schemes.</p>
                <p>This section will include:</p>
                <div className="empty-state-details">
                    <ul>
                        <li>Latest government subsidies and grants</li>
                        <li>Direct links to official application portals</li>
                        <li>Deadlines and important dates</li>
                    </ul>
                </div>
                <p>Check back soon for updates!</p>
            </div>
            
            <div id="schemes-container" className="schemes-grid" style={{ display: 'none' }}>
                {/* Future content */}
            </div>
        </div>
    );
}

window.renderReactSchemes = function () {
    const schemesContainer = document.getElementById('react-schemes-root');
    if (!schemesContainer) return;

    const root = ReactDOM.createRoot(schemesContainer);
    root.render(<SchemesApp />);
};

// Hook into the DOMContentLoaded event if we're using React manually to avoid racing with checkAuthStatus
document.addEventListener('DOMContentLoaded', () => {
    // Render History app if element is present
    if (document.getElementById('react-history-root')) {
        window.renderReactHistory();
    }
    // Render Schemes app if element is present
    if (document.getElementById('react-schemes-root')) {
        window.renderReactSchemes();
    }
});
