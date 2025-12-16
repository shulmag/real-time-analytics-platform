using QuickFix;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FixClientLite
{

    public interface ICompanyMessageCracker
    {
        /// <summary>Gets or sets the connection string key.</summary>
        string ConnectionStringKey { get; set; }

        /// <summary>The crack message.</summary>
        /// <param name="message">The message.</param>
        /// <param name="sessionID">The session id.</param>       
        void CrackMessage(Message message, SessionID sessionID);

        /// <summary>The on logon.</summary>
        /// <param name="sessionId">The session id.</param>       
        void OnLogon(SessionID sessionId);

        /// <summary>The on logout.</summary>
        /// <param name="sessionId">The session id.</param>        
        void OnLogout(SessionID sessionId);

        /// <summary>The send message.</summary>
        /// <param name="fixMessage">The fix message.</param>
        /// <param name="sessionId">The session id.</param>       
        void SendMessage(Message fixMessage, SessionID sessionId);

        /// <summary>
        /// simulate constructor
        /// </summary>
        /// <param name="stringToConfigureFrom">Connection string/File path...</param>
        void Initialize(string stringToConfigureFrom);

        void ToApp(Message message);

        void FromAdmin(Message message);
        void ToAdmin(Message message);

    }
}

